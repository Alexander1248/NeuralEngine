package ru.alexander.neuralengine.instructions;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

import static jcuda.driver.JCudaDriver.*;

public class Sum extends Instruction {
    public Sum(GpuExecutor executor) {
        super(executor);
    }

    @Override
    public String getInstructionName() {
        return "sum";
    }

    @Override
    public void compute(String... args) {
        Matrix out = getVariable(args[0]);
        Matrix in = getVariable(args[1]);

        int size = in.width() * in.height();

        CUdeviceptr ptr = new CUdeviceptr();
        cuMemAlloc(ptr, (long) Sizeof.FLOAT * size);
        cuMemcpyDtoD(ptr, in.pointer(), (long) Sizeof.FLOAT * size);

        int iterationSize = size - 1;
        do {
            iterationSize = (int) Math.ceil((double) iterationSize / 2);

            startGPUTask("mtxOperations.sum",
                    iterationSize, 1, 1,
                    Pointer.to(new int[] { iterationSize }),
                    Pointer.to(ptr)
            );
        } while (iterationSize > 1);



        float[] data = new float[size];
        cuMemcpyDtoH(Pointer.to(data), ptr, (long) Sizeof.FLOAT * size);
        data[0] += data[size - 1];
        cuMemcpyHtoD(out.pointer(), Pointer.to(data), Sizeof.FLOAT);
    }

    @Override
    public void addOutputVariable(String... args) {
        if (args.length < 2)
            throw new IllegalStateException("Instruction format error!");

        Matrix var = getVariable(args[1]);
        if (var == null)
            throw new IllegalStateException("Variable not exists!");

        if (hasVariable(args[0])
                && !variableSizeIsEqual(args[0], 1, 1))
            throw new IllegalStateException("Variable reformat error!");

        removeVariable(args[0]);
        addVariable(args[0], 1, 1);
    }
    public int[] getOutputVariableArgs() {
        return new int[] { 0 };
    }

    @Override
    public String documentation() {
        return """
                sum <out> <in>""";
    }
}
