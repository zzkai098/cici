import sys

from .cli.repl import run


def main():
    try:
        run()
    except RuntimeError as e:
        # Known, actionable setup failures get a clean line, not a traceback.
        print("cici: {}".format(e), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
