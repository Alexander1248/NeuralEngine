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
import ru.alexander.neuralengine.ioformats.ProjectIOFormat;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.*;

import static jcuda.driver.JCudaDriver.*;

public class GpuExecutor {
    // GPU System catalogs
    private final CUcontext context;
    private final Map<String, CUmodule> modules = new HashMap<>();
    private final Map<String, CUfunction> functions = new HashMap<>();

    // Compilation and execution instructions
    private final Map<String, String[]> scripts = new HashMap<>();
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

        for (int i = 0; i < code.length; i++) {
            String[] args = instructions.get(code[i].instruction())
                    .getOutputVariableArgs(code[i].args());
            for (int j = 0; j < args.length; j++)
                data.vars.remove(args[j]);
        }

        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < code.length; i++)
            builder.append(code[i]).append("\n");
        data.code = builder.toString();

        format.save(new FileOutputStream(filepath), data, this);
    }

    @Deprecated
    public void addScript(String scriptMask, String script) {
        int pos = scriptMask.indexOf(" ");
        if (pos == -1) pos = scriptMask.length();
        scripts.put(scriptMask.substring(0, pos), new String[] { scriptMask.substring(pos + 1), script });
    }
    @Deprecated
    public void loadScript(String filepath) throws IOException {
        FileInputStream reader = new FileInputStream(filepath);
        String data = new String(reader.readAllBytes()).replace("\r", "");
        reader.close();
        int pos = data.indexOf("\n");
        addScript(data.substring(0, pos), data.substring(pos));
    }
    @Deprecated
    public void removeScript(String scriptMask) {
        scripts.remove(scriptMask);
    }


    public void loadModule(String moduleName, String filepath) {
        CUmodule module = new CUmodule();
        cuModuleLoad(module, filepath);
        modules.put(moduleName, module);
    }
    public void loadModuleFromResources(String moduleName, String filepath) throws IOException {
        CUmodule module = new CUmodule();

        InputStream stream = getClass().getClassLoader().getResourceAsStream(filepath);
        if (stream == null)
            throw new IOException("File not found!");

        cuModuleLoadData(module, stream.readAllBytes());
        stream.close();

        modules.put(moduleName, module);
    }
    public void loadFunction(String functionName, String moduleName) {
        CUmodule module = modules.get(moduleName);
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
    public void loadDataInVariable(String name, float[] data) {
        Matrix mtx = vars.get(name);
        if (mtx == null)
            throw new IllegalStateException("Matrix not exists!");

        int size = mtx.width() * mtx.height();
        if (data.length != size)
            throw new IllegalStateException("Matrix size not compare to array size");
        cuMemcpyHtoD(vars.get(name).pointer(), Pointer.to(data), (long) Sizeof.FLOAT * size);
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
    public int[] getVariableSizes(String name) {
        Matrix mtx = vars.get(name);
        if (mtx == null)
            throw new IllegalStateException("Variable not exists!");
        return new int[] {
                mtx.width(),
                mtx.height()
        };
    }

    public void compile(String code) {
        Random random = new Random();
        LinkedList<String> segments = new LinkedList<>(List.of(code.replace("\r", "").split("\n")));

        List<InstructionDescription> instDesc = new ArrayList<>();
        while (!segments.isEmpty()) {
            String segment = segments.pollFirst();
            int pos = segment.indexOf("//");
            if (pos == -1) pos = segment.length();

            String instructionPart = segment.substring(0, pos);
            if (instructionPart.isEmpty()) continue;

            String[] instruction = instructionPart.split(" ");

            String[] script = scripts.get(instruction[0]);
            if (script != null)  {
                String codeBlock = script[1].replace("\r", "");
                String argsLine = script[0];

                // TODO: 13.11.2023 Fix after usage variables collision
                for (String key : vars.keySet()) {
                    byte[] bytes = new byte[16];
                    random.nextBytes(bytes);
                    String s;
                    do {
                        s = Base64.getEncoder().encodeToString(bytes);
                    } while (vars.containsKey(s));
                    if (argsLine.startsWith(key))
                        argsLine = argsLine.replaceFirst(key, "_" + s);
                    argsLine = argsLine.replace(" " + key, " _" + s);
                    codeBlock = codeBlock.replace(" " + key, " _" + s);
                }

                String[] args = argsLine.split(" ");
                for (int j = 0; j < args.length; j++)
                    codeBlock = codeBlock.replace(" " + args[j], " " + instruction[j + 1]);

                String[] codeFrags = codeBlock.split("\n");
                for (int j = codeFrags.length - 1; j >= 0; j--)
                    segments.addFirst(codeFrags[j]);
                continue;
            }

            InstructionDescription desc = new InstructionDescription(
                    instruction[0],
                    Arrays.copyOfRange(instruction, 1, instruction.length)
            );
            Instruction inst = instructions.get(desc.instruction());
            if (inst == null)
                throw new IllegalStateException("Instruction not exists:" + desc.instruction());

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
            StringBuilder node = new StringBuilder(" " + instruction.instruction);
            String[] args = instructions.get(instruction.instruction)
                    .getOutputVariableArgs(instruction.args);
            for (int j = 0; j < args.length; j++) {
                if (withSizes) {
                    Matrix mtx = vars.get(instruction.args[0]);
                    node.append(" \n ").append(mtx.width()).append(" ").append(mtx.height());
                }
                node.append(" \n ").append(instruction.args[0]).append(" ");
            }
            graph.addVertex(node.toString());

            for (int j = 1; j < instruction.args.length; j++) {
                String arg = instruction.args[j];
                List<String> list = graph.vertexSet().stream()
                        .filter(s -> s.endsWith(" " + arg + " ")).toList();
                String vert;
                if (list.isEmpty()) {
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
                else {
                    if (instruction.args[0].equals(arg))
                        vert = list.get(Math.max(0, list.size() - 2));
                    else  vert = list.get(list.size() - 1);
                }
                graph.addEdge(vert, node.toString());
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
        clearMemory();

        for (Map.Entry<String, CUmodule> entry : modules.entrySet())
            cuModuleUnload(entry.getValue());
        cuCtxDestroy(context);
    }

    Map<String, CUfunction> getFunctions() {
        return functions;
    }

    Map<String, Matrix> getVariables() {
        return vars;
    }

    @Override
    public boolean equals(Object object) {
        if (this == object) return true;
        if (object == null || getClass() != object.getClass()) return false;
        GpuExecutor that = (GpuExecutor) object;

        if (!Objects.equals(vars, that.vars)) return false;
        for (String s : vars.keySet())
            if (!Arrays.equals(getVariable(s), that.getVariable(s))) return false;

        return Objects.equals(instructions, that.instructions) && Arrays.equals(code, that.code);
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

        @Override
        public boolean equals(Object object) {
            if (this == object) return true;
            if (object == null || getClass() != object.getClass()) return false;
            InstructionDescription that = (InstructionDescription) object;
            return Objects.equals(instruction, that.instruction) && Arrays.equals(args, that.args);
        }

        @Override
        public int hashCode() {
            int result = Objects.hash(instruction);
            result = 31 * result + Arrays.hashCode(args);
            return result;
        }
    }
}
