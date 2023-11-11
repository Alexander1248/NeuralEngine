package ru.alexander.neuralengine.executor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

import java.util.Objects;

import static jcuda.driver.CUresult.CUDA_SUCCESS;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.cudaStreamSynchronize;

public abstract class Instruction {
    private final GpuExecutor executor;

    public Instruction(GpuExecutor executor) {
        this.executor = executor;
    }

    public abstract String getInstructionName();
    public abstract void compute(String... args);
    public abstract void addOutputVariable(String... args);
    public abstract String getOutputVariableArg(String... args);
    public abstract String documentation();


    protected boolean hasVariable(String name) {
        return executor.hasVariable(name);
    }
    protected boolean variableSizeIsEqual(String name, int width, int height) {
        Matrix mtx = executor.getVariableData(name);
        return mtx.width() == width && mtx.height() == height;
    }
    protected void removeVariable(String name) {
        executor.removeVariable(name);
    }
    protected Matrix getVariable(String name) {
        return executor.getVariableData(name);
    }
    protected void addVariable(String name, int width, int height) {
        executor.addVariable(name, width, height);
    }
    protected void startGPUTask(
            String function,
            int sx, int sy, int sz,
            Pointer... params) {
        CUfunction func = executor.getFunctions().get(function);
        if (func == null) throw new IllegalStateException("Function not exists!");

        int lx = Math.min(sx, 16);
        int gx = (int) Math.ceil((double) sx / 16);

        int ly = Math.min(sy, 16);
        int gy = (int) Math.ceil((double) sy / 16);

        int lz = Math.min(sz, 4);
        int gz = (int) Math.ceil((double) sz / 4);

        cuCtxSynchronize();
        int code = cuLaunchKernel(func,
                lx, ly, lz,
                gx, gy, gz,
                0, null, Pointer.to(params), null);
        if (code != CUDA_SUCCESS) {
            String[] pStr = new String[1];
            cuGetErrorString(code, pStr);
            System.out.println(pStr[0]);
        }
    }

    @Override
    public boolean equals(Object object) {
        if (this == object) return true;
        if (object == null || getClass() != object.getClass()) return false;
        Instruction that = (Instruction) object;
        return getInstructionName().equals(that.getInstructionName());
    }

    @Override
    public int hashCode() {
        return getInstructionName().hashCode();
    }
}
