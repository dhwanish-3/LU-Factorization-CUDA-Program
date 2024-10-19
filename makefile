# Makefile for CUDA program
# Compiler and flags
NVCC = nvcc
CFLAGS = -O3
# Program name (replace with your programfile)
PROGRAM = main.cu
OUTPUT = program
# Input and output files
INPUTS = indata50.txt indata80.txt indata100.txt indata200.txt indata500.txt
OUTPUTS = output50.txt output80.txt output100.txt output200.txt output500.txt
TIMINGS = timing50.txt timing80.txt timing100.txt timing200.txt timing500.txt
# Default
all: build
# Build the program
build:
	$(NVCC) $(CFLAGS) $(PROGRAM) -o $(OUTPUT)
# Run the program with each input and output file
run: build
	@./$(OUTPUT) indata50.txt output50.txt timing50.txt
	@./$(OUTPUT) indata80.txt output80.txt timing80.txt
	@./$(OUTPUT) indata100.txt output100.txt timing100.txt
	@./$(OUTPUT) indata200.txt output200.txt timing200.txt
	@./$(OUTPUT) indata500.txt output500.txt timing500.txt
# Clean the generated files
clean:
	rm -f $(OUTPUT) $(OUTPUTS) $(TIMINGS)
