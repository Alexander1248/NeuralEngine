package ru.alexander.neuralengine.instructions;

import jcuda.Pointer;
import jcuda.Sizeof;
import ru.alexander.neuralengine.executor.GpuExecutor;
import ru.alexander.neuralengine.executor.Instruction;
import ru.alexander.neuralengine.executor.Matrix;

import static jcuda.driver.JCudaDriver.cuMemcpyDtoD;

public class Sort extends Instruction {
    public Sort(GpuExecutor executor) {
        super(executor);
    }

    @Override
    public String getInstructionName() {
        return "sort";
    }

    @Override
    public void compute(String... args) {
        Matrix out = getVariable(args[0]);
        Matrix in = getVariable(args[1]);


        cuMemcpyDtoD(out.pointer(), in.pointer(), (long) Sizeof.FLOAT * in.width() * in.height());
        switch (args[2]) {
            case "x" -> {
                int size = in.width();
                int depth = 0;
                while (size != 1) {
                    size >>>= 1;
                    depth++;
                }
                for (int i = 1; i <= depth; i++)
                    for (int j = 0; j < i; j++)
                        startGPUTask("mtxOperations.sortX",
                                in.width(), in.height(), 1,
                                Pointer.to(new int[]{in.width()}),
                                Pointer.to(new int[]{in.height()}),
                                Pointer.to(new int[]{i}),
                                Pointer.to(new int[]{j}),
                                Pointer.to(out.pointer())
                        );
            }
            case "y" -> {
                int size = in.height();
                int depth = 0;
                while (size != 1) {
                    size >>>= 1;
                    depth++;
                }
                for (int i = 1; i <= depth; i++)
                    for (int j = 0; j < i; j++)
                        startGPUTask("mtxOperations.sortY",
                                in.width(), in.height(), 1,
                                Pointer.to(new int[]{in.width()}),
                                Pointer.to(new int[]{in.height()}),
                                Pointer.to(new int[]{i}),
                                Pointer.to(new int[]{j}),
                                Pointer.to(out.pointer())
                        );
            }
            case "linear" -> {
                int size = in.width() * in.height();
                int depth = 0;
                while (size != 1) {
                    size >>>= 1;
                    depth++;
                }
                for (int i = 1; i <= depth; i++)
                    for (int j = 0; j < i; j++)
                        startGPUTask("mtxOperations.sortX",
                                in.width() * in.height(), 1, 1,
                                Pointer.to(new int[]{in.width() * in.height()}),
                                Pointer.to(new int[]{1}),
                                Pointer.to(new int[]{i}),
                                Pointer.to(new int[]{j}),
                                Pointer.to(out.pointer())
                        );
            }
        }
    }

    @Override
    public void addOutputVariable(String... args) {
        if (args.length < 3)
            throw new IllegalStateException("Instruction format error!");

        if (!args[2].equals("x") && !args[2].equals("y") && !args[2].equals("linear"))
            throw new IllegalStateException("Wrong border type!");

        Matrix var = getVariable(args[1]);
        if (var == null)
            throw new IllegalStateException("Variable not exists!");

        int w, h;
        switch (args[2]) {
            case "x" -> {
                w = var.width();
                h = var.height();

                int size = w;
                while (size != 1) {
                    if (size % 2 != 0)
                        throw new IllegalStateException("Size is not power of 2!");
                    size >>>= 1;
                }
            }
            case "y" -> {
                w = var.width();
                h = var.height();

                int size = h;
                while (size != 1) {
                    if (size % 2 != 0)
                        throw new IllegalStateException("Size is not power of 2!");
                    size >>>= 1;
                }
            }
            case "linear" -> {
                w = var.width() * var.height();
                h = 1;

                int size = w;
                while (size != 1) {
                    if (size % 2 != 0)
                        throw new IllegalStateException("Size is not power of 2!");
                    size >>>= 1;
                }
            }
            default -> throw new IllegalStateException("Unexpected value: " + args[2]);
        }
        if (hasVariable(args[0])
                && !variableSizeIsEqual(args[0], w, h))
            throw new IllegalStateException("Variable reformat error!");

        removeVariable(args[0]);
        addVariable(args[0], w, h);
    }
    public int[] getOutputVariableArgs() {
        return new int[] { 0 };
    }

    @Override
    public String documentation() {
        return """
                sort <out> <in> <direction>""";
    }
}
