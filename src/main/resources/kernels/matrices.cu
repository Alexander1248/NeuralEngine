// Activation Functions
extern "C"
__global__ void relu(int width, int height, 
            float positiveCoefficient, float negativeCoefficient, 
            float* in, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pos = x + y * width;
        
        if (in[pos] >= 0) out[pos] = in[pos] * positiveCoefficient;
        else out[pos] = in[pos] * negativeCoefficient;
    }
}
extern "C"
__global__ void reluDer(int width, int height, 
            float positiveCoefficient, float negativeCoefficient, 
            float* in, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pos = x + y * width;
        
        if (in[pos] >= 0) out[pos] = positiveCoefficient;
        else out[pos] = negativeCoefficient;
    }
}


extern "C"
__global__ void sigmoid(int width, int height, 
            float force,
            float* in, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pos = x + y * width;
        
        out[pos] = 1.0 / (1.0 + exp(-force * in[pos]));
    }
}
extern "C"
__global__ void sigmoidDer(int width, int height, 
            float force,
            float* in, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pos = x + y * width;
        
        float val = exp(-force * in[pos]);
        out[pos] = force * val / pow(1.0 + val, 2);
    }
}


extern "C"
__global__ void tangent(int width, int height, 
            float force,
            float* in, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pos = x + y * width;
        
        out[pos] = 2.0 / (1.0 + exp(-force * in[pos])) - 1;
    }
}
extern "C"
__global__ void tangentDer(int width, int height, 
            float force,
            float* in, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pos = x + y * width;

         float val = exp(-force * in[pos]);
        out[pos] = 2.0 * force * val / pow(1.0 + val, 2);
    }
}


extern "C"
__global__ void softmax(int width, int height, 
            float force,
            float* in, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pos = x + y * width;
        
        int size = width * height;

        float sum = 0;
        for (int i = 0; i < size; i++)
            sum += exp(force * in[i]);

        out[pos] = exp(force * in[pos]) / sum;
    }
}
extern "C"
__global__ void softmaxDer(int width, int height, 
            float force,
            float* in, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pos = x + y * width;

        int size = width * height;
        
        float sum = 0;
        for (int i = 0; i < size; i++)
            sum += exp(force * in[i]);

        float e = exp(force * in[pos]);
        out[pos] =  force * e * (sum - e) / (sum * sum);
    }
}


// Unary operations
extern "C"
__global__ void transpose(int width, int height,
            float* in, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        out[x + y * width] = in[y + x * height];
    }
}


// Binary operations
extern "C"
__global__ void tensorAdd(int width, int height,
            float* in1, float* in2, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pos = x + y * width;
        out[pos] = in1[pos] + in2[pos];
    }
}
extern "C"
__global__ void tensorSub(int width, int height,
            float* in1, float* in2, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pos = x + y * width;
        out[pos] = in1[pos] - in2[pos];
    }
}

extern "C"
 __global__ void concatenateVertical(int width, int height1, int height2,
             float* in1, float* in2, float* out) {
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
             float* in1, float* in2, float* out) {
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
__global__ void tensorMul(int width, int height,
            float* in1, float* in2, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pos = x + y * width;
        out[pos] = in1[pos] * in2[pos];
    }
}

extern "C"
__global__ void tensorDiv(int width, int height,
            float* in1, float* in2, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pos = x + y * width;
        out[pos] = in1[pos] / in2[pos];
    }
}
 

extern "C"
__global__ void matrixMul(int w1h2, int height1, int width2,
            float* in1, float* in2, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width2 && y < height1) {
        int pos = x + y * width2;

        float sum = 0.0;
        for (int i = 0; i < w1h2; i++)
            sum += in1[i + y * w1h2] * in2[x + i * width2];

        out[pos] = sum;
    }
}


extern "C"
__global__ void matrixConvEmptyBorder(int width, int height, int mx, int my,
            float* in, float* matrix, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int hx = mx >> 1;
        int hy = my >> 1;

        float sum = 0;
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
            float* in, float* matrix, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int hx = mx >> 1;
        int hy = my >> 1;

        float sum = 0;
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
            float* in, float* matrix, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int hx = mx >> 1;
        int hy = my >> 1;

        float sum = 0;
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


extern "C"
__global__ void maxPooling(int width, int height,
            int rate,
            float* in, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int rx = x * rate;
        int ry = y * rate;
        int rw = width * rate;

        float val = -1e38;
        for (int dy = 0; dy < rate; dy++)
            for (int dx = 0; dx < rate; dx++)
                val = max(val, in[(rx + dx) + (ry + dy) * rw]);

        out[x + y * width] = val;
    }
}
extern "C"
__global__ void minPooling(int width, int height,
            int rate,
            float* in, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int rx = x * rate;
        int ry = y * rate;
        int rw = width * rate;

        float val = 1e38;
        for (int dy = 0; dy < rate; dy++)
            for (int dx = 0; dx < rate; dx++)
                val = min(val, in[(rx + dx) + (ry + dy) * rw]);

        out[x + y * width] = val;
    }
}
extern "C"
__global__ void avgPooling(int width, int height,
            int rate,
            float* in, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int rx = x * rate;
        int ry = y * rate;
        int rw = width * rate;

        float val = 0;
        for (int dy = 0; dy < rate; dy++)
            for (int dx = 0; dx < rate; dx++)
                val += in[(rx + dx) + (ry + dy) * rw];

        out[x + y * width] = val / (rate * rate);
    }
}