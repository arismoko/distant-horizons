// Capture ABI shared with render_bridge/hiz_proof.rs: twelve u32 header words,
// then two fixed-capacity reject lists and finally the row-major mip-0 words.
layout(std430) buffer HizProofBuffer {
    uint serial_lo;
    uint serial_hi;
    uint mips_written;
    uint copy_groups;
    uint trav_init_ran;
    uint trav_clears;
    uint trav_expected;
    uint trav_ran;
    uint trav_rejected;
    uint cmdgen_ran;
    uint cmdgen_rejected;
    uint reserved;
    uint payload[];
} hiz_proof;

const uint PROOF_REJECT_WORDS = 10u;
const uint PROOF_CAPACITY = uint(HIZ_PROOF_CAPACITY);
const uint PROOF_MIP0_OFFSET = 2u * PROOF_CAPACITY * PROOF_REJECT_WORDS;

void record_hiz_reject(bool cmdgen, uvec2 key, uint id, int level, float nearest,
                       float far_occluder, ivec2 lo, ivec2 hi) {
    uint slot;
    if (cmdgen) {
        slot = atomicAdd(hiz_proof.cmdgen_rejected, 1u);
    } else {
        slot = atomicAdd(hiz_proof.trav_rejected, 1u);
    }
    if (slot >= PROOF_CAPACITY) return;
    uint at = (slot + (cmdgen ? PROOF_CAPACITY : 0u)) * PROOF_REJECT_WORDS;
    hiz_proof.payload[at + 0u] = key.x;
    hiz_proof.payload[at + 1u] = key.y;
    hiz_proof.payload[at + 2u] = id;
    hiz_proof.payload[at + 3u] = uint(level);
    hiz_proof.payload[at + 4u] = floatBitsToUint(nearest);
    hiz_proof.payload[at + 5u] = floatBitsToUint(far_occluder);
    hiz_proof.payload[at + 6u] = uint(lo.x);
    hiz_proof.payload[at + 7u] = uint(lo.y);
    hiz_proof.payload[at + 8u] = uint(hi.x);
    hiz_proof.payload[at + 9u] = uint(hi.y);
}
