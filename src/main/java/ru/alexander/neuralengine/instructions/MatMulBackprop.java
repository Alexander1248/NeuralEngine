package ru.alexander.neuralengine.instructions;

import jcuda.Pointer;
import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

public class MatMulBackprop extends Instruction {
    public MatMulBackprop(GpuExecutor executor) {
        super(executor);
    }

    @Override
    public String getInstructionName() {
        return "matmul_backprop";
    }

    @Override
    public void compute(String... args) {
        Matrix prevError = getVariable(args[0]);
        Matrix in = getVariable(args[1]);
        Matrix currError = getVariable(args[2]);
        Matrix weights = getVariable(args[3]);

        startGPUTask("mtxOperations.matrixMulBackpropagationErrorTraversal",
                weights.height(), currError.height(), 1,
                Pointer.to(new int[]{weights.width()}),
                Pointer.to(new int[]{weights.height()}),
                Pointer.to(new int[]{currError.height()}),
                Pointer.to(currError.pointer()),
                Pointer.to(weights.pointer()),
                Pointer.to(prevError.pointer())
        );
        startGPUTask("mtxOperations.matrixMulBackpropagationWeightCorrection",
                currError.width(), in.width(), 1,
                Pointer.to(new int[]{currError.width()}),
                Pointer.to(new int[]{in.width()}),
                Pointer.to(new int[]{currError.height()}),
                Pointer.to(new float[]{Float.parseFloat(args[4])}),
                Pointer.to(in.pointer()),
                Pointer.to(currError.pointer()),
                Pointer.to(weights.pointer())
        );
    }

    @Override
    public void addOutputVariable(String... args) {
        if (args.length < 5)
            throw new IllegalStateException("Instruction format error!");

        Matrix in = getVariable(args[1]);
        Matrix currError = getVariable(args[2]);
        Matrix weights = getVariable(args[3]);

        if (in == null || currError == null || weights == null)
            throw new IllegalStateException("Variable not exists!");

        if (weights.width() != currError.width())
            throw new IllegalStateException("Wrong sizes for matrix multiplication!");

        if (currError.height() != in.height())
            throw new IllegalStateException("Wrong sizes for matrix multiplication!");

        try {
            Float.parseFloat(args[4]);
        } catch (Exception ex) {
            throw new IllegalStateException("Instruction format error!");
        }

        if (hasVariable(args[0])
                && !variableSizeIsEqual(args[0], weights.height(), currError.height()))
            throw new IllegalStateException("Variable reformat error!");

        removeVariable(args[0]);
        addVariable(args[0], weights.height(), currError.height());
    }
    public int[] getOutputVariableArgs() {
        return new int[] { 0, 3 };
    }

    @Override
    public String documentation() {
        return """
                matmul <prev error> <input> <curr error> <weights> <learning speed>""";
    }
}
