fn world_drawn(world: &mut WorldDrawn<'_>) {
    use distanthorizons::lod::SectionPos;
    use distanthorizons::mesh::{LodVertex, SectionMeshData};
    use distanthorizons::render_bridge::{
        COMPOSITE_WGSL, TERRAIN_WGSL, begin_render_frame, record_frame_stats,
    };
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    struct GpuSection {
        solid_vb: Option<wgpu::Buffer>,
        solid_ib: Option<wgpu::Buffer>,
        solid_indices: u32,
        fluid_vb: Option<wgpu::Buffer>,
        fluid_ib: Option<wgpu::Buffer>,
        fluid_indices: u32,
        min_y: f32,
        max_y: f32,
        quad_count: u32,
    }

    /// Extract the 4 side frustum planes (left, right, bottom, top) from `world.jittered`
    /// (column-major projection*view matrix). Each plane `[a, b, c, d]` accepts a camera-relative
    /// point `p` when `a*p.x + b*p.y + c*p.z + d >= 0`.
    fn extract_side_frustum_planes(mat: [f32; 16]) -> [[f32; 4]; 4] {
        // Row i of column-major mat is [mat[i], mat[4+i], mat[8+i], mat[12+i]].
        let row = |r: usize| [mat[r], mat[4 + r], mat[8 + r], mat[12 + r]];
        let r0 = row(0);
        let r1 = row(1);
        let r3 = row(3);
        let add = |a: [f32; 4], b: [f32; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
        let sub = |a: [f32; 4], b: [f32; 4]| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
        [
            add(r3, r0), // left:   x + w >= 0
            sub(r3, r0), // right:  w - x >= 0
            add(r3, r1), // bottom: y + w >= 0
            sub(r3, r1), // top:    w - y >= 0
        ]
    }

    /// Returns `true` if the camera-relative AABB `[min_x..max_x, min_y..max_y, min_z..max_z]`
    /// intersects the 4 side frustum planes (extended far plane is handled separately by distance).
    fn aabb_in_side_frustum(planes: &[[f32; 4]; 4], min_p: [f32; 3], max_p: [f32; 3]) -> bool {
        for p in planes {
            let px = if p[0] >= 0.0 { max_p[0] } else { min_p[0] };
            let py = if p[1] >= 0.0 { max_p[1] } else { min_p[1] };
            let pz = if p[2] >= 0.0 { max_p[2] } else { min_p[2] };
            if p[0] * px + p[1] * py + p[2] * pz + p[3] < 0.0 {
                return false;
            }
        }
        true
    }

    struct DhGpuState {
        color_format: wgpu::TextureFormat,
        size: [u32; 2],
        dh_color_view: wgpu::TextureView,
        dh_depth_view: wgpu::TextureView,
        terrain_uniform_buf: wgpu::Buffer,
        terrain_bind_group: wgpu::BindGroup,
        solid_pipeline: wgpu::RenderPipeline,
        fluid_pipeline: wgpu::RenderPipeline,
        instance_buf: wgpu::Buffer,
        instance_cap: usize,
        comp_uniform_buf: wgpu::Buffer,
        comp_layout: wgpu::BindGroupLayout,
        comp_pipeline: wgpu::RenderPipeline,
        sections: HashMap<SectionPos, GpuSection>,
        gpu_epoch: u64,
    }

    fn create_buffer_with_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
        usage: wgpu::BufferUsages,
        bytes: &[u8],
    ) -> wgpu::Buffer {
        let padded_len = ((bytes.len().max(4) + 3) & !3) as u64;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: padded_len,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if bytes.len() as u64 == padded_len {
            queue.write_buffer(&buf, 0, bytes);
        } else {
            let mut padded = vec![0u8; padded_len as usize];
            padded[..bytes.len()].copy_from_slice(bytes);
            queue.write_buffer(&buf, 0, &padded);
        }
        buf
    }

    fn create_offscreen_targets(
        device: &wgpu::Device,
        size: [u32; 2],
        color_format: wgpu::TextureFormat,
    ) -> (wgpu::TextureView, wgpu::TextureView) {
        let extent = wgpu::Extent3d {
            width: size[0].max(1),
            height: size[1].max(1),
            depth_or_array_layers: 1,
        };
        let color_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dh-offscreen-color"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dh-offscreen-depth"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        (
            color_tex.create_view(&wgpu::TextureViewDescriptor::default()),
            depth_tex.create_view(&wgpu::TextureViewDescriptor::default()),
        )
    }

    fn init_gpu_state(
        device: &wgpu::Device,
        size: [u32; 2],
        color_format: wgpu::TextureFormat,
        gpu_epoch: u64,
    ) -> DhGpuState {
        let (dh_color_view, dh_depth_view) = create_offscreen_targets(device, size, color_format);

        // Terrain pipeline
        let terrain_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dh-terrain-wgsl"),
            source: wgpu::ShaderSource::Wgsl(TERRAIN_WGSL.into()),
        });
        let terrain_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dh-terrain-uniforms"),
            size: 112,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let terrain_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dh-terrain-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let terrain_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dh-terrain-bg"),
            layout: &terrain_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: terrain_uniform_buf.as_entire_binding(),
            }],
        });
        let terrain_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dh-terrain-pl"),
            bind_group_layouts: &[Some(&terrain_bgl)],
            ..Default::default()
        });

        let vertex_attrs = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32,
                offset: 12,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32,
                offset: 16,
                shader_location: 2,
            },
        ];
        let instance_attrs = [wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 0,
            shader_location: 3,
        }];
        let buffers = [
            Some(wgpu::VertexBufferLayout {
                array_stride: LodVertex::STRIDE as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &vertex_attrs,
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &instance_attrs,
            }),
        ];

        let solid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dh-solid-pipeline"),
            layout: Some(&terrain_pl),
            vertex: wgpu::VertexState {
                module: &terrain_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &terrain_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let fluid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dh-fluid-pipeline"),
            layout: Some(&terrain_pl),
            vertex: wgpu::VertexState {
                module: &terrain_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &terrain_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let instance_cap = 1024;
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dh-section-instances"),
            size: (instance_cap * 16) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Fullscreen sky-masked composite + atmospheric fog pipeline
        let comp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dh-composite-wgsl"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_WGSL.into()),
        });
        let comp_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dh-composite-uniforms"),
            size: 48,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let comp_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dh-composite-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let comp_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dh-composite-pl"),
            bind_group_layouts: &[Some(&comp_layout)],
            ..Default::default()
        });
        let comp_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dh-composite-pipeline"),
            layout: Some(&comp_pl),
            vertex: wgpu::VertexState {
                module: &comp_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &comp_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        DhGpuState {
            color_format,
            size,
            dh_color_view,
            dh_depth_view,
            terrain_uniform_buf,
            terrain_bind_group,
            solid_pipeline,
            fluid_pipeline,
            instance_buf,
            instance_cap,
            comp_uniform_buf,
            comp_layout,
            comp_pipeline,
            sections: HashMap::new(),
            gpu_epoch,
        }
    }

    fn upload_mesh(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &SectionMeshData,
    ) -> GpuSection {
        let (solid_vb, solid_ib, solid_indices) = if !mesh.solid_indices.is_empty() {
            let vb = create_buffer_with_bytes(
                device,
                queue,
                "dh-solid-vb",
                wgpu::BufferUsages::VERTEX,
                &mesh.solid_vertex_bytes,
            );
            let mut idx_bytes = Vec::with_capacity(mesh.solid_indices.len() * 4);
            for &i in &mesh.solid_indices {
                idx_bytes.extend_from_slice(&i.to_le_bytes());
            }
            let ib = create_buffer_with_bytes(
                device,
                queue,
                "dh-solid-ib",
                wgpu::BufferUsages::INDEX,
                &idx_bytes,
            );
            (Some(vb), Some(ib), mesh.solid_indices.len() as u32)
        } else {
            (None, None, 0)
        };

        let (fluid_vb, fluid_ib, fluid_indices) = if !mesh.fluid_indices.is_empty() {
            let vb = create_buffer_with_bytes(
                device,
                queue,
                "dh-fluid-vb",
                wgpu::BufferUsages::VERTEX,
                &mesh.fluid_vertex_bytes,
            );
            let mut idx_bytes = Vec::with_capacity(mesh.fluid_indices.len() * 4);
            for &i in &mesh.fluid_indices {
                idx_bytes.extend_from_slice(&i.to_le_bytes());
            }
            let ib = create_buffer_with_bytes(
                device,
                queue,
                "dh-fluid-ib",
                wgpu::BufferUsages::INDEX,
                &idx_bytes,
            );
            (Some(vb), Some(ib), mesh.fluid_indices.len() as u32)
        } else {
            (None, None, 0)
        };

        GpuSection {
            solid_vb,
            solid_ib,
            solid_indices,
            fluid_vb,
            fluid_ib,
            fluid_indices,
            min_y: mesh.min_y,
            max_y: mesh.max_y,
            quad_count: mesh.quad_count,
        }
    }

    static GPU: OnceLock<Mutex<Option<DhGpuState>>> = OnceLock::new();

    if world.size[0] == 0 || world.size[1] == 0 {
        return;
    }

    let cam = [world.camera.x, world.camera.y, world.camera.z];
    let (params, uploads, epoch) = begin_render_frame(cam);
    if !params.enabled {
        record_frame_stats(0, 0);
        return;
    }

    let mut guard = GPU.get_or_init(|| Mutex::new(None)).lock().unwrap();
    let recreate = match guard.as_ref() {
        None => true,
        Some(st) => st.color_format != world.color_format,
    };
    if recreate {
        *guard = Some(init_gpu_state(
            world.device,
            world.size,
            world.color_format,
            epoch,
        ));
    }
    let state = guard.as_mut().unwrap();
    if state.size != world.size {
        let (cv, dv) = create_offscreen_targets(world.device, world.size, world.color_format);
        state.dh_color_view = cv;
        state.dh_depth_view = dv;
        state.size = world.size;
    }
    if state.gpu_epoch != epoch {
        state.sections.clear();
        state.gpu_epoch = epoch;
    }

    // Upload newly meshed sections to GPU.
    for mesh in &uploads {
        let gpu_sec = upload_mesh(world.device, world.queue, mesh);
        state.sections.insert(mesh.pos, gpu_sec);
    }

    // Build frustum-culled draw list and per-section camera-relative instance offsets.
    let mat_cols = world.jittered.to_cols_array();
    let frustum_planes = extract_side_frustum_planes(mat_cols);
    let mut draw_list: Vec<SectionPos> = Vec::with_capacity(params.visible_sections.len());
    let mut instance_bytes: Vec<u8> = Vec::with_capacity(params.visible_sections.len() * 16);
    let mut total_quads: u64 = 0;

    for &pos in &params.visible_sections {
        let Some(sec) = state.sections.get(&pos) else {
            continue;
        };
        if sec.solid_indices == 0 && sec.fluid_indices == 0 {
            continue;
        }
        let (ox, oz) = pos.block_origin();
        let span = pos.block_span() as f32;
        let rel_x = (ox as f64 - world.camera.x) as f32;
        let rel_y = (-world.camera.y) as f32;
        let rel_z = (oz as f64 - world.camera.z) as f32;
        let min_p = [rel_x, sec.min_y + rel_y, rel_z];
        let max_p = [rel_x + span, sec.max_y + rel_y, rel_z + span];
        if !aabb_in_side_frustum(&frustum_planes, min_p, max_p) {
            continue;
        }
        let cell_sz = pos.cell_size() as f32;
        instance_bytes.extend_from_slice(&rel_x.to_le_bytes());
        instance_bytes.extend_from_slice(&rel_y.to_le_bytes());
        instance_bytes.extend_from_slice(&rel_z.to_le_bytes());
        instance_bytes.extend_from_slice(&cell_sz.to_le_bytes());
        total_quads += sec.quad_count as u64;
        draw_list.push(pos);
    }

    if draw_list.is_empty() {
        record_frame_stats(0, 0);
        return;
    }

    if draw_list.len() > state.instance_cap {
        let new_cap = draw_list.len().next_power_of_two();
        state.instance_buf = world.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dh-section-instances"),
            size: (new_cap * 16) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        state.instance_cap = new_cap;
    }
    world
        .queue
        .write_buffer(&state.instance_buf, 0, &instance_bytes);

    let dh_far = (params.config.render_distance_blocks() * 1.4).max(1024.0);
    let near_dh = 8.0f32;
    let clip_dist = (params.vanilla_far_blocks * params.config.overdraw_prevention)
        .clamp(0.0, params.vanilla_far_blocks * 0.85);
    let fade_band = (params.vanilla_far_blocks * 0.25).clamp(8.0, 48.0);

    // Pack Terrain FrameUniforms (112 bytes)
    let mut u_bytes = [0u8; 112];
    for (i, f) in mat_cols.iter().enumerate() {
        u_bytes[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
    }
    let clip_and_light = [near_dh, dh_far, clip_dist, params.sun_brightness];
    for (i, f) in clip_and_light.iter().enumerate() {
        u_bytes[64 + i * 4..68 + i * 4].copy_from_slice(&f.to_le_bytes());
    }
    let noise_and_fog = [
        params.config.noise_intensity,
        fade_band,
        params.fog_color[0],
        params.fog_color[1],
    ];
    for (i, f) in noise_and_fog.iter().enumerate() {
        u_bytes[80 + i * 4..84 + i * 4].copy_from_slice(&f.to_le_bytes());
    }
    let is_hdr = matches!(
        world.color_format,
        wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rg11b10Ufloat
    );
    let flags = [
        u32::from(params.config.shading_enabled),
        u32::from(params.config.noise_enabled),
        u32::from(params.config.fade_enabled),
        u32::from(is_hdr),
    ];
    for (i, v) in flags.iter().enumerate() {
        u_bytes[96 + i * 4..100 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    world
        .queue
        .write_buffer(&state.terrain_uniform_buf, 0, &u_bytes);

    // Pass 1: Render Distant Horizons solid + translucent LOD terrain into offscreen targets.
    {
        let mut pass = world
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dh-offscreen-terrain-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &state.dh_color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: params.fog_color[0] as f64,
                            g: params.fog_color[1] as f64,
                            b: params.fog_color[2] as f64,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &state.dh_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

        pass.set_bind_group(0, &state.terrain_bind_group, &[]);
        pass.set_vertex_buffer(1, state.instance_buf.slice(..));

        // Opaque solid pass (front-to-back)
        pass.set_pipeline(&state.solid_pipeline);
        for (idx, pos) in draw_list.iter().enumerate() {
            let sec = &state.sections[pos];
            if sec.solid_indices > 0 {
                if let (Some(vb), Some(ib)) = (&sec.solid_vb, &sec.solid_ib) {
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    let inst = idx as u32;
                    pass.draw_indexed(0..sec.solid_indices, 0, inst..(inst + 1));
                }
            }
        }

        // Translucent fluid pass (back-to-front over solid terrain depth)
        pass.set_pipeline(&state.fluid_pipeline);
        for (idx, pos) in draw_list.iter().enumerate().rev() {
            let sec = &state.sections[pos];
            if sec.fluid_indices > 0 {
                if let (Some(vb), Some(ib)) = (&sec.fluid_vb, &sec.fluid_ib) {
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    let inst = idx as u32;
                    pass.draw_indexed(0..sec.fluid_indices, 0, inst..(inst + 1));
                }
            }
        }
    }

    // Pass 2: Composite DH terrain + atmospheric horizon fog into vanilla's sky pixels.
    let fog_end = params.config.render_distance_blocks().max(256.0);
    let fog_start = (fog_end * 0.45).max(params.vanilla_far_blocks * 0.9);
    let mut comp_bytes = [0u8; 48];
    for (i, f) in params.fog_color.iter().enumerate() {
        comp_bytes[i * 4..i * 4 + 4].copy_from_slice(&f.to_le_bytes());
    }
    let comp_params = [near_dh, dh_far, fog_start, fog_end];
    for (i, f) in comp_params.iter().enumerate() {
        comp_bytes[16 + i * 4..20 + i * 4].copy_from_slice(&f.to_le_bytes());
    }
    let comp_flags = [
        u32::from(params.config.fog_enabled),
        u32::from(is_hdr),
        0u32,
        0u32,
    ];
    for (i, v) in comp_flags.iter().enumerate() {
        comp_bytes[32 + i * 4..36 + i * 4].copy_from_slice(&v.to_le_bytes());
    }
    world
        .queue
        .write_buffer(&state.comp_uniform_buf, 0, &comp_bytes);

    let comp_bg = world.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dh-composite-bg"),
        layout: &state.comp_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state.comp_uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&state.dh_color_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&state.dh_depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(world.depth),
            },
        ],
    });

    {
        let mut pass = world
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dh-composite-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: world.color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
        pass.set_pipeline(&state.comp_pipeline);
        pass.set_bind_group(0, &comp_bg, &[]);
        pass.draw(0..3, 0..1);
    }

    record_frame_stats(draw_list.len() as u64, total_quads);
}
