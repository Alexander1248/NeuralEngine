package ru.alexander.neuralengine.instructions;

import jcuda.Pointer;
import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

public class ConvBackprop extends Instruction {
    public ConvBackprop(GpuExecutor executor) {
        super(executor);
    }

    @Override
    public String getInstructionName() {
        return "conv_backprop";
    }

    @Override
    public void compute(String... args) {
        Matrix prevError = getVariable(args[0]);
        Matrix in = getVariable(args[1]);
        Matrix currError = getVariable(args[1]);
        Matrix mtx = getVariable(args[2]);


        switch (args[3]) {
            case "empty" -> {
                startGPUTask("mtxOperations.matrixConvEmptyBorderBackpropagationErrorTraversal",
                        in.width(), in.height(), 1,
                        Pointer.to(new int[]{in.width()}),
                        Pointer.to(new int[]{in.height()}),
                        Pointer.to(new int[]{mtx.width()}),
                        Pointer.to(new int[]{mtx.height()}),
                        Pointer.to(currError.pointer()),
                        Pointer.to(mtx.pointer()),
                        Pointer.to(prevError.pointer())
                );
                startGPUTask("mtxOperations.matrixConvEmptyBorderBackpropagationWeightCorrection",
                        in.width(), in.height(), 1,
                        Pointer.to(new int[]{in.width()}),
                        Pointer.to(new int[]{in.height()}),
                        Pointer.to(new int[]{mtx.width()}),
                        Pointer.to(new int[]{mtx.height()}),
                        Pointer.to(new float[]{Float.parseFloat(args[4])}),
                        Pointer.to(in.pointer()),
                        Pointer.to(currError.pointer()),
                        Pointer.to(mtx.pointer())
                );
            }
            case "extend" -> {
                startGPUTask("mtxOperations.matrixConvExtendBorderBackpropagationErrorTraversal",
                        in.width(), in.height(), 1,
                        Pointer.to(new int[]{in.width()}),
                        Pointer.to(new int[]{in.height()}),
                        Pointer.to(new int[]{mtx.width()}),
                        Pointer.to(new int[]{mtx.height()}),
                        Pointer.to(currError.pointer()),
                        Pointer.to(mtx.pointer()),
                        Pointer.to(prevError.pointer())
                );
                startGPUTask("mtxOperations.matrixConvExtendBorderBackpropagationWeightCorrection",
                        in.width(), in.height(), 1,
                        Pointer.to(new int[]{in.width()}),
                        Pointer.to(new int[]{in.height()}),
                        Pointer.to(new int[]{mtx.width()}),
                        Pointer.to(new int[]{mtx.height()}),
                        Pointer.to(new float[]{Float.parseFloat(args[4])}),
                        Pointer.to(in.pointer()),
                        Pointer.to(currError.pointer()),
                        Pointer.to(mtx.pointer())
                );
            }
            case "repeat" -> {
                startGPUTask("mtxOperations.matrixConvRepeatBorderBackpropagationErrorTraversal",
                        in.width(), in.height(), 1,
                        Pointer.to(new int[]{in.width()}),
                        Pointer.to(new int[]{in.height()}),
                        Pointer.to(new int[]{mtx.width()}),
                        Pointer.to(new int[]{mtx.height()}),
                        Pointer.to(currError.pointer()),
                        Pointer.to(mtx.pointer()),
                        Pointer.to(prevError.pointer())
                );
                startGPUTask("mtxOperations.matrixConvRepeatBorderBackpropagationWeightCorrection",
                        in.width(), in.height(), 1,
                        Pointer.to(new int[]{in.width()}),
                        Pointer.to(new int[]{in.height()}),
                        Pointer.to(new int[]{mtx.width()}),
                        Pointer.to(new int[]{mtx.height()}),
                        Pointer.to(new float[]{Float.parseFloat(args[4])}),
                        Pointer.to(in.pointer()),
                        Pointer.to(currError.pointer()),
                        Pointer.to(mtx.pointer())
                );
            }
        }
    }

    @Override
    public void addOutputVariable(String... args) {
        if (args.length < 6)
            throw new IllegalStateException("Instruction format error!");

        if (!args[5].equals("empty") && !args[5].equals("extend") && !args[5].equals("repeat"))
            throw new IllegalStateException("Wrong border type!");

        Matrix var1 = getVariable(args[1]);
        Matrix var2 = getVariable(args[2]);
        Matrix var3 = getVariable(args[3]);

        if (var1 == null || var2 == null || var3 == null)
            throw new IllegalStateException("Variable not exists!");

        try {
            Float.parseFloat(args[4]);
        } catch (Exception ex) {
            throw new IllegalStateException("Instruction format error!");
        }

        if (hasVariable(args[0])
                && !variableSizeIsEqual(args[0], var1.width(), var1.height()))
            throw new IllegalStateException("Variable reformat error!");

        removeVariable(args[0]);
        addVariable(args[0], var1.width(), var1.height());
    }
    public int[] getOutputVariableArgs() {
        return new int[] { 0, 3 };
    }

    @Override
    public String documentation() {
        return """
                conv_backprop <prev error> <input> <curr error> <matrix> <learning speed> <border type>""";
    }
}
