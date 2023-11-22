package ru.alexander.neuralengine.instructions;

import jcuda.Pointer;
import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

public class Set extends Instruction {
    public Set(GpuExecutor executor) {
        super(executor);
    }

    @Override
    public String getInstructionName() {
        return "set";
    }

    @Override
    public void compute(String... args) {
        Matrix out = getVariable(args[0]);

        if (isDoublePrecision()) {
            double value;
            try {
                value = Double.parseDouble(args[3]);
            } catch (Exception ex) {
                value = getVariableMtxDouble(args[3])[0];
            }

            startGPUTask("mtxOperations.set",
                    out.width(), out.height(), 1,
                    Pointer.to(new int[]{out.width()}),
                    Pointer.to(new int[]{out.height()}),
                    Pointer.to(new double[]{value}),
                    Pointer.to(out.pointer())
            );
        }
        else {
            float value;
            try {
                value = Float.parseFloat(args[3]);
            } catch (Exception ex) {
                value = getVariableMtxFloat(args[3])[0];
            }

            startGPUTask("mtxOperations.set",
                    out.width(), out.height(), 1,
                    Pointer.to(new int[]{out.width()}),
                    Pointer.to(new int[]{out.height()}),
                    Pointer.to(new float[]{value}),
                    Pointer.to(out.pointer())
            );
        }
    }

    @Override
    public void addOutputVariable(String... args) {
        if (args.length < 4)
            throw new IllegalStateException("Instruction format error!");

        int width, height;
        try {
            width = Integer.parseInt(args[1]);
            height = Integer.parseInt(args[2]);
        } catch (Exception ex) {
            throw new IllegalStateException("Instruction format error!");
        }

        try {
            Float.parseFloat(args[3]);
        } catch (Exception ex) {
            try {
                if (isDoublePrecision())
                    getVariableMtxDouble(args[3]);
                else
                    getVariableMtxFloat(args[3]);
            } catch (Exception ex1) {
                throw new IllegalStateException("Instruction format error!");
            }
        }

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
                set <out> <width> <height> <value>""";
    }
}
