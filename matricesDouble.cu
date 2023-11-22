// Unary operations
extern "C"
__global__ void transpose(int width, int height,
            double* in, double* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        out[x + y * width] = in[y + x * height];
    }
}

extern "C"
__global__ void flipX(int width, int height,
            double* in, double* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        out[x + y * width] = in[(width - 1 - x) + y * width];
    }
}
extern "C"
__global__ void flipY(int width, int height,
            double* in, double* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        out[x + y * width] = in[x + (height - 1 - y) * width];
    }
}

extern "C"
__global__ void rotate90(int width, int height,
            double* in, double* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        out[x + y * width] = in[(height - 1 - y) + x * height];
    }
}
extern "C"
__global__ void rotate180(int width, int height,
            double* in, double* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        out[x + y * width] = in[(width - 1 - x) + (height - 1 - y) * width];
    }
}
extern "C"
__global__ void rotate270(int width, int height,
            double* in, double* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        out[x + y * width] = in[y + (width - 1 - x) * height];
    }
}

extern "C"
__global__ void set(int width, int height, double value,
            double* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        out[x + y * width] = value;
    }
}

extern "C"
__global__ void sum(int size, double* mtx) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < size) {
        mtx[x] += mtx[x + size];
        mtx[x + size] = 0;
    }
}


// Binary operations
extern "C"
__global__ void tensorAdd(int width, int height,
            double* in1, double* in2, double* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pos = x + y * width;
        out[pos] = in1[pos] + in2[pos];
    }
}
extern "C"
__global__ void tensorSub(int width, int height,
            double* in1, double* in2, double* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pos = x + y * width;
        out[pos] = in1[pos] - in2[pos];
    }
}

extern "C"
 __global__ void concatenateVertical(int width, int height1, int height2,
             double* in1, double* in2, double* out) {
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;

     if (x < width && y < height1 + height2) {
         int pos = x + y * width;

        if (y < height1) out[pos] = in1[pos];
        else {
            y -= height1;
            out[pos] = in2[x + y * width];
        }
     }
}

extern "C"
 __global__ void concatenateHorizontal(int width1, int width2, int height,
             double* in1, double* in2, double* out) {
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;

     if (x < width1 + width2 && y < height) {
         int pos = x + y * (width1 + width2);

        if (x < width1) out[pos] = in1[x + y * width1];
        else {
            x -= width1;
            out[pos] = in2[x + y * width2];
        }
     }
}


extern "C"
__global__ void mul(int width, int height, float value,
            double* in, double* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pos = x + y * width;
        out[pos] = in[pos] * value;
    }
}
extern "C"
__global__ void tensorMul(int width, int height,
            double* in1, double* in2, double* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pos = x + y * width;
        out[pos] = in1[pos] * in2[pos];
    }
}
extern "C"
__global__ void matrixMul(int w1h2, int height1, int width2,
            double* in1, double* in2, double* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width2 && y < height1) {
        double sum = 0.0;
        for (int i = 0; i < w1h2; i++)
            sum += in1[i + y * w1h2] * in2[x + i * width2];

        out[x + y * width2] = sum;
    }
}

extern "C"
__global__ void tensorDiv(int width, int height,
            double* in1, double* in2, double* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pos = x + y * width;
        out[pos] = in1[pos] / in2[pos];
    }
}


extern "C"
__global__ void matrixConvEmptyBorder(int width, int height, int mx, int my,
            double* in, double* matrix, double* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int hx = mx >> 1;
        int hy = my >> 1;

        double sum = 0;
        for (int dy = 0; dy < mx; dy++) {
            int py = y + dy - hy;
            if (py < 0) continue;
            if (py >= height) continue;

            for (int dx = 0; dx < mx; dx++) {
               int px = x + dx - hx;
                if (px < 0) continue;
                if (px >= width) continue;

                sum += in[px + py * width] * matrix[dx + dy * mx];
            }
        }
        out[x + y * width] = sum;
    }
}

extern "C"
__global__ void matrixConvExtendBorder(int width, int height, int mx, int my,
            double* in, double* matrix, double* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int hx = mx >> 1;
        int hy = my >> 1;

        double sum = 0;
        for (int dy = 0; dy < mx; dy++) {
            int py = max(0, min(height - 1, y + dy - hy));

            for (int dx = 0; dx < mx; dx++) {
                int px = max(0, min(width - 1, x + dx - hx));

                sum += in[px + py * width] * matrix[dx + dy * mx];
            }
        }
        out[x + y * width] = sum;
    }
}

extern "C"
__global__ void matrixConvRepeatBorder(int width, int height, int mx, int my,
            double* in, double* matrix, double* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int hx = mx >> 1;
        int hy = my >> 1;

        double sum = 0;
        for (int dy = 0; dy < mx; dy++) {
            int py = y + dy - hy;
            if (py < 0) py += height;
            if (py >= height) py -= height;

            for (int dx = 0; dx < mx; dx++) {
                int px = x + dx - hx;
                if (px < 0) px += width;
                if (px >= width) px -= width;

                sum += in[px + py * width] * matrix[dx + dy * mx];
            }
        }
        out[x + y * width] = sum;
    }
}