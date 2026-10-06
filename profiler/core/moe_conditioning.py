"""Deterministic full-position weight reuse control, independent of timings."""
import math


def expert_conditioning(contract, active, minimum_iterations):
    local = contract['local_experts']
    if (type(local) is not int or local < 1 or type(active) is not int
            or not 0 <= active <= local or type(minimum_iterations) is not int
            or minimum_iterations < 1):
        raise ValueError('Invalid expert conditioning geometry')
    policy = contract['expert_conditioning']
    if policy['schema'] != 'rotated-weight-cycles-v2':
        raise ValueError('Unknown expert conditioning protocol')
    if (type(policy['capacity_multiple']) is not int or policy['capacity_multiple'] < 1
            or not 0 < policy['quality_relative_tolerance'] < 1
            or policy['representative'] != 'higher_bank_control'
            or policy['warmup_complete_cycles'] != 3):
        raise ValueError('Unsupported expert conditioning policy')
    weight_bytes = contract['expert_weight_bytes']
    l2_bytes = policy['l2_bytes']
    if any(type(v) is not int or v < 1 for v in (weight_bytes,l2_bytes)):
        raise ValueError('Measured weight size and actual GPU cache capacity are required')
    offsets = list(range(0,local,active)) if active else [0]
    banks = policy.get('minimum_banks', 2)
    if type(banks) is not int or banks < 1 or banks & (banks-1):
        raise ValueError('The minimum independent weight-bank count must be a positive power of two')
    if active:
        # Rotation visits the complete local weight bank, not just the first
        # A experts. This is a conditioning guard, never a latency equation.
        while banks*weight_bytes < policy['capacity_multiple']*l2_bytes:
            banks *= 2
    if not active:
        banks = 1
    control = 2*banks if active else 1
    cycles = [len(offsets)*b for b in (banks,control)]
    common = math.lcm(*cycles)
    count = math.ceil(minimum_iterations/common)*common
    # Both arms cover whole cycles, but must also launch exactly the same
    # amount of warm-up and timed work. Otherwise bank count is confounded
    # with execution count and the length of uninterrupted GPU activity.
    forwards = [count,count]
    return dict(offsets=offsets,bank_counts=[banks,control],cycles=cycles,
                forward_counts=forwards,warmup_forwards=3*count,selected_arm=1)
