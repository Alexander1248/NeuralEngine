package ru.alexander.neuralengine.executor;

import jcuda.driver.CUdeviceptr;

public record Matrix(int width, int height, CUdeviceptr pointer) {
}
