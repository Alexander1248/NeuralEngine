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
        float vp1 = 1.0 + val;
        out[pos] = force * val / (vp1 * vp1);
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
        float vp1 = 1.0 + val;
        out[pos] = 2.0 * force * val / (vp1 * vp1);
    }
}


extern "C"
__global__ void softmax(int width, int height, 
            float force,
            float* in, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int yw = y * width;
        int pos = x + yw;

        float sum = 0;
        for (int i = 0; i < width; i++)
            sum += exp(force * in[i + yw]);

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
        int yw = y * width;
        int pos = x + yw;
        
        float sum = 0;
        for (int i = 0; i < width; i++)
            sum += exp(force * in[i + yw]);

        float e = exp(force * in[pos]);
        out[pos] =  force * e * (sum - e) / (sum * sum);
    }
}


extern "C"
__global__ void matrixMulBackpropagationErrorTraversal(int w1w2, int height1, int height2,
            float* currError, float* weigts, float* prevError) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < height2 && y < height1) {
        float sum = 0.0;
        for (int i = 0; i < w1w2; i++)
            sum += currError[i + y * height1] * weigts[x + i * w1w2];

        prevError[x + y * height2] = sum;
    }
}

extern "C"
__global__ void matrixMulBackpropagationWeightCorrection(int h1h2, int width1, int width2, float learningSpeed,
            float* input, float* error, float* weightsDelta) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width2 && y < width1) {
        float sum = 0.0;
        for (int i = 0; i < h1h2; i++)
            sum += input[y + i * width1] * error[x + i * width2];

        weightsDelta[x + y * width2] = sum;
    }
}



extern "C"
__global__ void matrixConvEmptyBorderBackpropagationErrorTraversal(int width, int height, int mx, int my,
            float* currError, float* matrix, float* prevError) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int hx = mx >> 1;
        int hy = my >> 1;

        float sum = 0;
        for (int dy = 0; dy < mx; dy++) {
            int py = y + my - 1 - dy - hy;
            if (py < 0) continue;
            if (py >= height) continue;

            for (int dx = 0; dx < mx; dx++) {
               int px = x + mx - 1 - dx - hx;
                if (px < 0) continue;
                if (px >= width) continue;

                sum += currError[px + py * width] * matrix[dx + dy * width];
            }
        }
        prevError[x + y * width] = sum;
    }
}
extern "C"
__global__ void matrixConvEmptyBorderBackpropagationWeightCorrection(int width, int height, int mx, int my, float ls,
            float* input, float* error, float* matrixDelta) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < mx && y < my) {
        int hx = mx >> 1;
        int hy = my >> 1;

        int sx = x - hx;
        int sy = y - hy;

        float sum = 0;
        for (int dy = 0; dy < height; dy++) {
            int py = dy - sy;
            if (py < 0) continue;
            if (py >= height) continue;
            for (int dx = 0; dx < width; dx++) {
                int px = dx - sx;
                if (px < 0) continue;
                if (px >= width) continue;
                sum += error[dx + dy * width] * input[px + py * width];
            }
        }
        matrixDelta[x + y * width] = sum * ls;
    }
}


extern "C"
__global__ void matrixConvExtendBorderBackpropagationErrorTraversal(int width, int height, int mx, int my,
            float* in, float* matrix, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int hx = mx >> 1;
        int hy = my >> 1;

        float sum = 0;
        for (int dy = 0; dy < mx; dy++) {
            int py = max(0, min(height - 1, y + my - 1 - dy - hy));

            for (int dx = 0; dx < mx; dx++) {
                int px = max(0, min(width - 1, x + mx - 1 - dx - hx));

                sum += in[px + py * width] * matrix[dx + dy * mx];
            }
        }
        out[x + y * width] = sum;
    }
}
extern "C"
__global__ void matrixConvExtendBorderBackpropagationWeightCorrection(int width, int height, int mx, int my, float ls,
            float* input, float* error, float* matrixDelta) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < mx && y < my) {
        int hx = mx >> 1;
        int hy = my >> 1;

        int sx = x - hx;
        int sy = y - hy;

        float sum = 0;
        for (int dy = 0; dy < height; dy++)
            for (int dx = 0; dx < width; dx++) {
                int px = max(0, min(width - 1, dx - sx));
                int py = max(0, min(height - 1, dy - sy));
                sum += error[dx + dy * width] * input[px + py * width];
            }
        matrixDelta[x + y * width] = sum * ls;
    }
}


extern "C"
__global__ void matrixConvRepeatBorderBackpropagationErrorTraversal(int width, int height, int mx, int my,
            float* in, float* matrix, float* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int hx = mx >> 1;
        int hy = my >> 1;

        float sum = 0;
        for (int dy = 0; dy < mx; dy++) {
            int py = y + my - 1 - dy - hy;
            if (py < 0) py += height;
            if (py >= height) py -= height;

            for (int dx = 0; dx < mx; dx++) {
                int px = x + mx - 1 - dx - hx;
                if (px < 0) px += width;
                if (px >= width) px -= width;

                sum += in[px + py * width] * matrix[dx + dy * mx];
            }
        }
        out[x + y * width] = sum;
    }
}
extern "C"
__global__ void matrixConvRepeatBorderBackpropagationWeightCorrection(int width, int height, int mx, int my, float ls,
            float* input, float* error, float* matrixDelta) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < mx && y < my) {
        int hx = mx >> 1;
        int hy = my >> 1;

        int sx = x - hx;
        int sy = y - hy;

        float sum = 0;
        for (int dy = 0; dy < height; dy++) {
            int py = dy - sy;
            if (py < 0) py += height;
            if (py >= height) py -= height;
            for (int dx = 0; dx < width; dx++) {
                int px = dx - sx;
                if (px < 0) px += width;
                if (px >= width) px -= width;
                sum += error[dx + dy * width] * input[px + py * width];
            }
        }
        matrixDelta[x + y * width] = sum * ls;
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
__global__ void maxminPoolingBackpropagation(int width, int height, int rate,
            float* in, float* out, 
            float* errorNext, float* errorPrev) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int rx = x / rate;
        int ry = y / rate;
        int rw = width / rate;

        errorPrev[x + y * width] = 0;
        if (abs(in[x + y * width] - out[rx + ry * rw]) < 1e-5) 
            errorPrev[x + y * width] = errorNext[rx + ry * width];
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
extern "C"
__global__ void avgPoolingBackpropagation(int width, int height, int rate,
            float* in, float* out, 
            float* errorNext, float* errorPrev) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int rx = x / rate;
        int ry = y / rate;
        int rw = width / rate;

        errorPrev[x + y * width] = errorNext[rx + ry * width] * in[x + y + width] / (out[rx + ry * rw] * rate * rate);
    }
}