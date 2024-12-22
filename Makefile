# Compiler and flags
CC = gcc
CFLAGS = -O3 -march=native -ffast-math -funroll-loops
LDFLAGS = -lm -flto

# Target and source
TARGET = a.out
SRC = main.c

# Build target
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) -o $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean up
clean:
	rm -f $(TARGET)