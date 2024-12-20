# Makefile for rasterizer program

# Compiler and flags
CC = gcc
CFLAGS = -O3
LDFLAGS = -lm

# Target and source
TARGET = a.out
SRC = rasterizer.c

# Build target
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) -o $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean up
clean:
	rm -f $(TARGET)