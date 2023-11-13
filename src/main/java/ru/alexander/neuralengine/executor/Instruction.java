package ru.alexander.neuralengine.executor;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;

public abstract class Instruction {
    private final GpuExecutor executor;

    public Instruction(GpuExecutor executor) {
        this.executor = executor;
    }

    public abstract String getInstructionName();
    public abstract void compute(String... args);
    public abstract void addOutputVariable(String... args);
    public abstract int[] getOutputVariableArgs();
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
    protected float[] getVariableMtx(String name) {
        return executor.getVariable(name);
    }
    protected void loadDataInVariable(String name, float[] data) {
        executor.loadDataInVariable(name, data);
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

        int mod = sx / (int) Math.ceil((double) sx / 1023) + 1;
        int lx = Math.min(sx, mod);
        int gx = (int) Math.ceil((double) sx / mod);

        mod = sy / (int) Math.ceil((double) sy / 1023) + 1;
        int ly = Math.min(sy, mod);
        int gy = (int) Math.ceil((double) sy / mod);

        mod = sz / (int) Math.ceil((double) sz / 63) + 1;
        int lz = Math.min(sz, mod);
        int gz = (int) Math.ceil((double) sz / mod);

        cuLaunchKernel(func,
                lx, ly, lz,
                gx, gy, gz,
                0, null, Pointer.to(params), null);
        cuCtxSynchronize();
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
