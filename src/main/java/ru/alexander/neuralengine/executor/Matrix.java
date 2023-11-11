package ru.alexander.neuralengine.executor;

import jcuda.driver.CUdeviceptr;

import java.util.Objects;

public record Matrix(int width, int height, CUdeviceptr pointer) {

    @Override
    public boolean equals(Object object) {
        if (this == object) return true;
        if (object == null || getClass() != object.getClass()) return false;
        Matrix matrix = (Matrix) object;
        return width == matrix.width && height == matrix.height;
    }

    @Override
    public int hashCode() {
        return Objects.hash(width, height);
    }
}
