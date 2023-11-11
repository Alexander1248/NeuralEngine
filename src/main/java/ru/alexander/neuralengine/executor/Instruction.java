package ru.alexander.neuralengine.executor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

import static jcuda.driver.JCudaDriver.*;

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
        int gx = (sx >>> 4) + 1;

        int ly = Math.min(sy, 16);
        int gy = (sy >>> 4) + 1;

        int lz = Math.min(sz, 4);
        int gz = (sz >>> 2) + 1;

        cuCtxSynchronize();
        cuLaunchKernel(func,
                lx, ly, lz,
                gx, gy, gz,
                0, null, Pointer.to(params), null);
    }
}
