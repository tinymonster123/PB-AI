from src.load_models import download_tinyllama
from src.quantize import quantize


def main():
    download_tinyllama()
    quantize()


if __name__ == "__main__":
    main()
