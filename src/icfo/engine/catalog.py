"""Which effect keys the engine understands, and how.

Every effect on a placed champion falls in exactly one class:

* DAMAGE      multiplies champion damage; scored when it reaches the carry
* AMPLIFIER   scales another upgrade's amount (``buff_upgrade`` family).
              Plain buffs to the same upgrade ADD together; ``_mult`` /
              ``_multiplicative`` variants multiply (the game defines explicit
              multiplicative and "additively increases" keys).
* MODIFIER    rewrites another upgrade's data/targets (``change_upgrade_data``...)
* PARAM       bookkeeping that only holds numbers for other keys (``pre_stack``)
* NON_DAMAGE  known not to change anyone's damage multiplier (healing, gold...)
* INTRINSIC   changes only the source's own attacks and doesn't depend on the
              formation, so it can't change which formation is best (for a
              fixed carry). Ignored, but only when it really targets just the
              source and has no formation-dependent stack count.
* GOLD        gold find; scored into a separate "gold" channel
* SPEED       progress speed (spawns, quests, time scale, Briv); collected for
              the gem-farm model
* anything else is UNMODELLED and reported.
"""

DAMAGE_KEYS = frozenset({
    "hero_dps_multiplier_mult",
    "global_dps_multiplier_mult",
    "hero_dps_mult_per_tagged_crusader_mult",
    "hero_dps_mult_per_target_crusader",
    "hero_dps_mult_per_target_tag_crusader_mult_amount_before",
})

AMPLIFIER_KEYS = frozenset({
    "buff_upgrade",
    "buff_upgrades",
    "buff_upgrade_per_any_tagged_crusader_mult",
    "buff_upgrade_per_any_tagged_crusader",
    "buff_upgrade_per_any_crusader_where_mult",
    "buff_upgrade_by_target_tag_mult",
    "buff_upgrade_add_flat_amount",
    "buff_upgrade_multiplicative",
    "buff_upgrades_multiplicative",
    "buff_upgrade_add_then_mult",
    "buff_upgrade_by_tag_mult",
    "buff_upgrade_by_tag_additive",
    "buff_upgrade_per_crusader",
    "buff_upgrade_per_unique_race",
    "buff_upgrade_per_target_crusader",
    "buff_upgrade_per_target_crusader_mult",
})

MODIFIER_KEYS = frozenset({
    "change_upgrade_data",
    "change_upgrade_targets",
    "add_target_to_upgrade",
    "force_add_target_to_upgrade",
})

PARAM_KEYS = frozenset({
    "pre_stack",
    "pre_stack_amount",
    "stacks_data_binder_safe",
    "base_amount",
    "just_for_stacks",
    "do_nothing",
})

NON_DAMAGE_KEYS = frozenset({
    "health_add",
    "health_mult",
    "heal",
    "healing_add",
    "healing_add_mult",
    "increase_health_by_source_percent",
    "grant_temporary_hp_with_cooldown",
    "damage_reduction",
    "appear_dead",
    "scavenge_items",
    "add_hero_tags",
    "force_allow_hero",
    "animation_synced_overlay",
    "disallow_roaming_familiar_ult_trigger",
    "mirror_image_preference",
    "add_attack_stun",
    "monster_speed_reduce",
})

INTRINSIC_KEYS = frozenset({
    "reduce_attack_cooldown",
    "set_base_crit_chance",
    "buff_ultimate",
    "set_ultimate_attack",
    "change_base_attack",
    "base_attack_deal_bonus_damage",
    "increase_damage_against_monster",
    "increase_damage_against_monster_armor_and_hits",
    "monster_with_tag_more_damage",
    "chance_on_attack_to_reset_attack_cooldown",
    "add_crit_effect",
    "chance_attack_adds_dot",
    "add_attack_aoe_targets",
    "add_attack_targets",
    "add_monster_hit_effects",
    "deal_bud_damage_when_hit",
    "favored_foe",
})

# stack functions the engine can count from the formation alone
COUNTED_STACK_FUNCS = frozenset({
    "per_hero_attribute",
    "per_crusader",
    "per_hero",
    "per_tagged_crusader_mult",
    "adjacent_champions",
    "per_unique_race",
    "per_upgrade_targets",
})

# how a stack count combines with the amount
SUPPORTED_AMOUNT_FUNCS = frozenset({None, "mult", "add", "if", "if_not", "mult_with_self"})

# Gold find: scored into the "gold" channel (formation-wide, not carry-targeted)
GOLD_KEYS = frozenset({
    "gold_multiplier_mult",
    "gold_mult_per_tagged_crusader_mult",
    "gold_mult_per_tagged_crusader",
})

# Progress speed: collected into the "speed" channel for the gem-farm model
SPEED_KEYS = frozenset({
    "increase_monster_spawn_time_mult",
    "decrease_monster_spawn_time_mult",
    "decrease_monster_spawn_delay_mult",
    "spawn_additional_monsters",
    "simultaneous_monster_spawn_chance_mult",
    "simultaneous_monster_spawn_chance_add",
    "chance_reduce_quest_requirement",
    "reduce_quest_requirements_free_play",
    "chance_multiply_monster_quest_rewards",
    "chance_multiply_monster_quest_rewards_new",
    "killing_blow_quest_mult",
    "time_scale",
    "time_scale_when_not_attacked",
    "area_transition_time_scale",
    "briv_unnatural_haste",
    "briv_unnatural_haste_stack_consume_percent_override",
    # gems per boss (gem objective)
    "increase_boss_gems_percent",
    "increase_boss_gems",
    "ellywick_deck_of_many_things",
})
