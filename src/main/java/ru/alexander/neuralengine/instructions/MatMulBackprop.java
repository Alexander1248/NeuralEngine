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
        Matrix weightsDelta = getVariable(args[1]);
        Matrix in = getVariable(args[2]);
        Matrix currError = getVariable(args[3]);
        Matrix weights = getVariable(args[4]);

        startGPUTask("neuralOperations.matrixMulBackpropagationErrorTraversal",
                weights.height(), currError.width(), 1,
                Pointer.to(new int[]{currError.height()}),
                Pointer.to(new int[]{currError.width()}),
                Pointer.to(new int[]{weights.height()}),
                Pointer.to(currError.pointer()),
                Pointer.to(weights.pointer()),
                Pointer.to(prevError.pointer())
        );
        startGPUTask("neuralOperations.matrixMulBackpropagationWeightCorrection",
                currError.width(), in.width(), 1,
                Pointer.to(new int[]{in.height()}),
                Pointer.to(new int[]{in.width()}),
                Pointer.to(new int[]{currError.width()}),
                Pointer.to(new float[]{Float.parseFloat(args[5])}),
                Pointer.to(in.pointer()),
                Pointer.to(currError.pointer()),
                Pointer.to(weightsDelta.pointer())
        );
    }

    @Override
    public void addOutputVariable(String... args) {
        if (args.length < 5)
            throw new IllegalStateException("Instruction format error!");

        Matrix in = getVariable(args[2]);
        Matrix currError = getVariable(args[3]);
        Matrix weights = getVariable(args[4]);

        if (currError.width() != weights.width())
            throw new IllegalStateException("Wrong sizes for matrix multiplication!");

        if (in.height() != currError.height())
            throw new IllegalStateException("Wrong sizes for matrix multiplication!");

        try {
            Float.parseFloat(args[5]);
        } catch (Exception ex) {
            throw new IllegalStateException("Instruction format error!");
        }

        if (hasVariable(args[0])
                && !variableSizeIsEqual(args[0], weights.height(), currError.height()))
            throw new IllegalStateException("Variable reformat error!");


        if (hasVariable(args[1])
                && !variableSizeIsEqual(args[1], currError.width(), in.width()))
            throw new IllegalStateException("Variable reformat error!");

        removeVariable(args[0]);
        addVariable(args[0], weights.height(), currError.height());

        removeVariable(args[1]);
        addVariable(args[1], currError.width(), in.width());
    }
    public int[] getOutputVariableArgs() {
        return new int[] { 0, 1 };
    }

    @Override
    public String documentation() {
        return """
                matmul_backprop <prev error> <weights delta> <input> <curr error> <weights> <learning speed>""";
    }
}
