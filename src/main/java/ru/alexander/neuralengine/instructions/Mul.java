package ru.alexander.neuralengine.instructions;

import jcuda.Pointer;
import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

public class Mul extends Instruction {
    public Mul(GpuExecutor executor) {
        super(executor);
    }

    @Override
    public String getInstructionName() {
        return "mul";
    }

    @Override
    public void compute(String... args) {
        Matrix out = getVariable(args[0]);
        Matrix in1 = getVariable(args[1]);

        try {
            startGPUTask("mtxOperations.mul",
                    in1.width(), in1.height(), 1,
                    Pointer.to(new int[]{in1.width()}),
                    Pointer.to(new int[]{in1.height()}),
                    Pointer.to(new float[]{ Float.parseFloat(args[2]) }),
                    Pointer.to(in1.pointer()),
                    Pointer.to(out.pointer())
            );
        } catch (Exception e) {
            Matrix in2 = getVariable(args[2]);

            startGPUTask("mtxOperations.tensorMul",
                    in1.width(), in1.height(), 1,
                    Pointer.to(new int[]{in1.width()}),
                    Pointer.to(new int[]{in1.height()}),
                    Pointer.to(in1.pointer()),
                    Pointer.to(in2.pointer()),
                    Pointer.to(out.pointer())
            );
        }
    }

    @Override
    public void addOutputVariable(String... args) {
        if (args.length < 3)
            throw new IllegalStateException("Instruction format error!");

        Matrix var1 = getVariable(args[1]);

        try {
            Float.parseFloat(args[2]);
        } catch (Exception e) {
            Matrix var2 = getVariable(args[2]);

            if (var1.width() != var2.width())
                throw new IllegalStateException("Widths not equal!");

            if (var1.height() != var2.height())
                throw new IllegalStateException("Heights not equal!");
        }

        if (hasVariable(args[0])
                && !variableSizeIsEqual(args[0], var1.width(), var1.height()))
            throw new IllegalStateException("Variable reformat error!");

        removeVariable(args[0]);
        addVariable(args[0], var1.width(), var1.height());
    }
    public int[] getOutputVariableArgs() {
        return new int[] { 0 };
    }

    @Override
    public String documentation() {
        return """
                mul <out> <in1> <in2>""";
    }
}
