"""Debug script to show generated decoder code for a message definition.

Usage:
    python scripts/show_generated_code.py "uint8 blank\nfloat64[] arr"
"""

import argparse

from mcap_ros2_support_fast._dynamic_decoder import DecoderGeneratorFactory
from mcap_ros2_support_fast._dynamic_encoder import EncoderGeneratorFactory
from mcap_ros2_support_fast._planner import generate_plans, optimize_plan


# ruff: noqa: T201
def main() -> None:
    parser = argparse.ArgumentParser(description="Show generated code for a message definition.")
    parser.add_argument("message_definition", help="The message definition to generate code for.")
    parser.add_argument(
        "--no-optimize", action="store_true", help="Disable optimization of the plan."
    )
    args = parser.parse_args()

    msg_def = args.message_definition

    # Interpret escape sequences like \n
    msg_def = msg_def.encode().decode("unicode_escape")

    print("Message definition:")
    print("=" * 60)
    print(msg_def)
    print("=" * 60)
    print()

    plan = generate_plans("custom_type/TestMsg", msg_def)
    if not args.no_optimize:
        plan = optimize_plan(plan)

    # Generate the decoder code without executing it
    factory = DecoderGeneratorFactory(plan, comments=True)
    target_type_name = f"decoder_{plan[0].__name__}_main"
    code = factory.generate_decoder_code(target_type_name)

    print("Generated decoder code:")
    print("=" * 60)
    print(code)
    print("=" * 60)

    # Generate the encoder code without executing it
    factory = EncoderGeneratorFactory(plan, comments=True)
    target_type_name = f"encoder_{plan[0].__name__}_main"
    code = factory.generate_encoder_code(target_type_name)

    print("Generated encoder code:")
    print("=" * 60)
    print(code)
    print("=" * 60)


if __name__ == "__main__":
    main()
