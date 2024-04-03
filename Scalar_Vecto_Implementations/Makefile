CC = nvcc
CFLAGS = -I/usr/include/opencv4
LDFLAGS = -L/usr/lib/x86_64-linux-gnu -lopencv_core -lopencv_highgui -lopencv_imgcodecs

TARGET_SCA = grayscale_filter_sca
TARGET_VEC = grayscale_filter_vec

sca: $(TARGET_SCA)

vec: $(TARGET_VEC)

all: $(TARGET_SCA) $(TARGET_VEC)

$(TARGET_SCA): grayscale_filter_sca.cu
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(TARGET_VEC): grayscale_filter_vec.cu
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET_SCA) $(TARGET_VEC)
