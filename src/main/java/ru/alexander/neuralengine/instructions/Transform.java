package ru.alexander.neuralengine.instructions;

import jcuda.Sizeof;
import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

import static jcuda.driver.JCudaDriver.cuMemcpyDtoD;

public class Transform extends Instruction {
    public Transform(GpuExecutor executor) {
        super(executor);
    }

    @Override
    public String getInstructionName() {
        return "transform";
    }

    @Override
    public void compute(String... args) {
        Matrix out = getVariable(args[0]);
        Matrix in = getVariable(args[1]);
        cuMemcpyDtoD(out.pointer(), in.pointer(), (long) Sizeof.FLOAT * in.width() * in.height());
    }

    @Override
    public void addOutputVariable(String... args) {
        if (args.length < 4)
            throw new IllegalStateException("Instruction format error!");

        Matrix var = getVariable(args[1]);
        if (var == null)
            throw new IllegalStateException("Variable not exists!");

        int width, height;
        try {
            width = Integer.parseInt(args[2]);
            height = Integer.parseInt(args[3]);
        } catch (Exception ex) {
            throw new IllegalStateException("Instruction format error!");
        }

        if (width * height != var.width() * var.height())
            throw new IllegalStateException("Sizes not equal!");

        if (hasVariable(args[0])
                && !variableSizeIsEqual(args[0], width, height))
            throw new IllegalStateException("Variable reformat error!");

        removeVariable(args[0]);
        addVariable(args[0], width, height);
    }
    public int[] getOutputVariableArgs() {
        return new int[] { 0 };
    }

    @Override
    public String documentation() {
        return """
                transform <out> <in> <width> <height>""";
    }
}
