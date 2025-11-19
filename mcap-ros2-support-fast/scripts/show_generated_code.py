"""Debug script to show generated decoder code for a message definition.

Usage:
    python scripts/show_generated_code.py "uint8 blank\nfloat64[] arr"
    python scripts/show_generated_code.py "uint8 blank\nfloat64[] arr" --endianness both
    python scripts/show_generated_code.py "uint8 blank\nfloat64[] arr" --endianness big
"""

import argparse

from mcap_ros2_support_fast._dynamic_decoder import DecoderGeneratorFactory
from mcap_ros2_support_fast._dynamic_encoder import EncoderGeneratorFactory
from mcap_ros2_support_fast._planner import generate_plans, optimize_plan


# ruff: noqa: T201
def show_code_for_endianness(plan: tuple, endianness: str, label: str) -> None:
    """Generate and display code for a specific endianness."""
    print(f"\n{'=' * 80}")
    print(f"{label} ({endianness}-endian)")
    print("=" * 80)

    # Generate the decoder code
    decoder_factory = DecoderGeneratorFactory(plan, comments=True, endianness=endianness)
    target_type_name = f"decoder_{plan[0].__name__}_main"
    decoder_code = decoder_factory.generate_decoder_code(target_type_name)

    print("\nDecoder code:")
    print("-" * 80)
    print(decoder_code)

    # Generate the encoder code
    encoder_factory = EncoderGeneratorFactory(plan, comments=True, endianness=endianness)
    target_type_name = f"encoder_{plan[0].__name__}_main"
    encoder_code = encoder_factory.generate_encoder_code(target_type_name)

    print("\nEncoder code:")
    print("-" * 80)
    print(encoder_code)

    # Show struct patterns to verify no conflicts
    print("\nStruct patterns:")
    print("-" * 80)
    print("Decoder patterns:")
    for pattern, var_name in decoder_factory.struct_patterns.items():
        print(f"  {var_name}: {pattern}")
    print("Encoder patterns:")
    for pattern, var_name in encoder_factory.struct_patterns.items():
        print(f"  {var_name}: {pattern}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Show generated code for a message definition.")
    parser.add_argument("message_definition", help="The message definition to generate code for.")
    parser.add_argument(
        "--no-optimize", action="store_true", help="Disable optimization of the plan."
    )
    parser.add_argument(
        "--endianness",
        choices=["little", "big", "both"],
        default="both",
        help="Endianness to generate code for (default: both)",
    )
    args = parser.parse_args()

    msg_def = args.message_definition

    # Interpret escape sequences like \n
    msg_def = msg_def.encode().decode("unicode_escape")

    print("Message definition:")
    print("=" * 80)
    print(msg_def)
    print("=" * 80)

    plan = generate_plans("custom_type/TestMsg", msg_def)
    if not args.no_optimize:
        plan = optimize_plan(plan)

    if args.endianness in ("little", "both"):
        show_code_for_endianness(plan, "<", "LITTLE-ENDIAN")

    if args.endianness in ("big", "both"):
        show_code_for_endianness(plan, ">", "BIG-ENDIAN")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
