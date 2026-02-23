from src.load_models import ensure_tinyllama_local
from src.quantize import quantize


def main():
    ensure_tinyllama_local()
    quantize()


if __name__ == "__main__":
    main()
