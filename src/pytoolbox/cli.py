import argparse
from pathlib import Path
from cv_helpers import ImageProcessor

class CLIApplication:
    """Handles command-line argument parsing and tool execution."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="PyToolbox CLI Utility")
        self._setup_arguments()

    def _setup_arguments(self) -> None:
        self.parser.add_argument("--input", type=str, required=True, help="Path to input image")
        self.parser.add_argument("--output", type=str, required=True, help="Path to output processed image")
        self.parser.add_argument("--blur", action="store_true", help="Apply Gaussian blur")

    def run(self) -> None:
        args = self.parser.parse_args()
        processor = ImageProcessor(args.input)
        
        processor.convert_to_grayscale()
        if args.blur:
            processor.apply_gaussian_blur()
            
        processor.save_output(args.output)
        print(f"Successfully processed {args.input} -> {args.output}")

if __name__ == "__main__":
    app = CLIApplication()
    app.run()