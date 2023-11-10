package ru.alexander.neuralengine.executor;

import com.mxgraph.layout.hierarchical.mxHierarchicalLayout;
import com.mxgraph.layout.mxIGraphLayout;
import com.mxgraph.util.mxCellRenderer;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import org.jgraph.graph.DefaultEdge;
import org.jgrapht.ext.JGraphXAdapter;
import org.jgrapht.graph.DefaultDirectedGraph;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.List;


import static jcuda.driver.JCudaDriver.*;

public class GpuExecutor {
    // GPU System catalogs
    private final CUcontext context;
    private final Map<String, CUmodule> scripts = new HashMap<>();
    private final Map<String, CUfunction> functions = new HashMap<>();

    // Compilation and execution instructions
    private final Map<String, Instruction> instructions = new HashMap<>();

    // Code parameters
    private final Map<String, Matrix> vars = new HashMap<>();
    private InstructionDescription[] code;

    public GpuExecutor() {
        setExceptionsEnabled(true);

        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        context = new CUcontext();
        cuCtxCreate(context, 0, device);
    }

    public void loadProject(String filepath, ProjectIOFormat format) throws IOException {
        int pos = filepath.indexOf(".");
        if (pos == -1) throw new IOException("File without format");
        if (!format.isRightFormat(filepath.substring(pos + 1)))
            throw new IOException("Wrong file format!");

        clearMemory();
        format.load(new FileInputStream(filepath), this);
    }
    public void saveProject(String filepath, ProjectIOFormat format) throws IOException {
        int pos = filepath.indexOf(".");
        if (pos == -1) throw new IOException("File without format!");
        if (!format.isRightFormat(filepath.substring(pos + 1)))
            throw new IOException("Wrong file format!");

        ProjectIOFormat.Data data = new ProjectIOFormat.Data();
        data.instructions = instructions;
        data.vars = new HashMap<>(vars);

        for (int i = 0; i < code.length; i++)
            data.vars.remove(instructions.get(code[i].instruction())
                    .getOutputVariableArg(code[i].args()));

        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < code.length; i++)
            builder.append(code[i]).append("\n");
        data.code = builder.toString();

        format.save(new FileOutputStream(filepath), data, this);
    }


    public void loadModule(String moduleName, String filepath) {
        CUmodule module = new CUmodule();
        cuModuleLoad(module, filepath);
        scripts.put(moduleName, module);
    }
    public void loadModuleFromResources(String moduleName, String filepath) throws IOException {
        CUmodule module = new CUmodule();

        InputStream stream = getClass().getClassLoader().getResourceAsStream(filepath);
        if (stream == null)
            throw new IOException("File not found!");

        cuModuleLoadData(module, stream.readAllBytes());
        stream.close();

        scripts.put(moduleName, module);
    }
    public void loadFunction(String functionName, String moduleName) {
        CUmodule module = scripts.get(moduleName);
        if (module == null)
            throw new IllegalStateException("Module not exists!");

        CUfunction func = new CUfunction();
        cuModuleGetFunction(func, module, functionName);

        functions.put(moduleName + "." + functionName, func);
    }

    public void addInstruction(Instruction instruction) {
        instructions.put(instruction.getInstructionName(), instruction);
    }
    public boolean hasInstruction(String instructionName) {
        return instructions.containsKey(instructionName);
    }


    public void addVariable(String name, int width, int height) {
        CUdeviceptr ptr = new CUdeviceptr();
        cuMemAlloc(ptr, (long) Sizeof.FLOAT * width * height);
        vars.put(name, new Matrix(width, height, ptr));
    }
    public void addVariable(String name, int width, int height, float[] data) {
        if (data.length != width * height)
            throw new IllegalStateException("Matrix size not compare to array size");
        addVariable(name, width, height);
        cuMemcpyHtoD(vars.get(name).pointer(), Pointer.to(data), (long) Sizeof.FLOAT * width * height);
    }
    Matrix getVariableData(String name) {
        Matrix mtx = vars.get(name);
        if (mtx == null)
            throw new IllegalStateException("Variable not exists: " + name);
        return mtx;
    }
    boolean hasVariable(String name) {
        return vars.containsKey(name);
    }
    void removeVariable(String name) {
        Matrix mtx = vars.get(name);
        if (mtx == null) return;

        cuMemFree(mtx.pointer());
        vars.remove(name);
    }
    public float[] getVariable(String name) {
        Matrix mtx = vars.get(name);
        if (mtx == null)
            throw new IllegalStateException("Variable not exists!");
        float[] data = new float[mtx.width() * mtx.height()];
        cuMemcpyDtoH(Pointer.to(data), mtx.pointer(), (long) Sizeof.FLOAT * data.length);
        return data;
    }

    public void compile(String code) {
        String[] segments = code.replace("\r", "").split("\n");

        List<InstructionDescription> instDesc = new ArrayList<>();
        for (int i = 0; i < segments.length; i++) {
            int pos = segments[i].indexOf("//");
            if (pos == -1) pos = segments[i].length();

            String instructionPart = segments[i].substring(0, pos);
            if (instructionPart.isEmpty()) continue;

            String[] instruction = instructionPart.split(" ");
            InstructionDescription desc = new InstructionDescription(
                    instruction[0],
                    Arrays.copyOfRange(instruction, 1, instruction.length)
            );
            Instruction inst = instructions.get(desc.instruction());
            if (inst == null)
                throw new IllegalStateException("Instruction not exists:" + desc.instruction());

            removeVariable(inst.getOutputVariableArg(desc.args()));
            inst.addOutputVariable(desc.args());
            instDesc.add(desc);
        }
        this.code = instDesc.toArray(InstructionDescription[]::new);
    }

    public void compute() {
        for (int i = 0; i < code.length; i++) {
            Instruction instruction = instructions.get(code[i].instruction());
            instruction.compute(code[i].args());
        }
    }
    public BufferedImage visualize(boolean withSizes) {
        DefaultDirectedGraph<String, DefaultEdge> graph = new DefaultDirectedGraph<>(DefaultEdge.class);

        for (int i = 0; i < code.length; i++) {
            InstructionDescription instruction = code[i];
            String node =  " " + instruction.instruction;
            if (withSizes) {
                Matrix mtx = vars.get(instruction.args[0]);
                node += " \n " + mtx.width() + " " + mtx.height();
            }
            node += " \n " + instruction.args[0] + " ";
            graph.addVertex(node);

            for (int j = 1; j < instruction.args.length; j++) {
                String arg = instruction.args[j];
                String vert = graph.vertexSet().stream()
                        .filter(s -> s.endsWith(" " + arg + " ")).findAny().orElse(null);
                if (vert == null) {
                    String name = "";
                    if (vars.containsKey(arg)) {
                        name = " mtx \n";
                        if (withSizes) {
                            Matrix mtx = vars.get(arg);
                            name += " " + mtx.width() + " " + mtx.height() + " \n";
                        }
                    }
                    name += " " + arg + " ";

                    graph.addVertex(name);
                    vert = name;
                }
                graph.addEdge(vert, node);
            }
        }


        JGraphXAdapter<String, DefaultEdge> adapter = new JGraphXAdapter<>(graph);
        mxIGraphLayout layout = new mxHierarchicalLayout(adapter);
        layout.execute(adapter.getDefaultParent());
        return mxCellRenderer.createBufferedImage(adapter, null, 2, Color.white, true, null);
    }

    public String getDocumentation() {
        StringBuilder builder = new StringBuilder();
        instructions.forEach((name, instruction) ->
                builder.append(instruction.documentation()).append("\n"));
        return builder.toString();
    }

    public void clearMemory() {
        vars.forEach((name, mtx) -> cuMemFree(mtx.pointer()));
        vars.clear();
    }
    public void close() {
        for (Map.Entry<String, CUmodule> entry : scripts.entrySet())
            cuModuleUnload(entry.getValue());
        cuCtxDestroy(context);

        clearMemory();
    }

    Map<String, CUfunction> getFunctions() {
        return functions;
    }

    Map<String, Matrix> getVariables() {
        return vars;
    }

    private record InstructionDescription(String instruction, String[] args) {

        @Override
        public String toString() {
            StringBuilder builder = new StringBuilder();
            builder.append(instruction);
            for (int i = 0; i < args.length; i++)
                builder.append(" ").append(args[i]);
            return builder.toString();
        }
    }
}
