package in.mcxiv.proc;

import com.google.auto.service.AutoService;
import com.squareup.javapoet.ClassName;
import com.squareup.javapoet.JavaFile;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.TypeSpec;
import in.mcxiv.annotations.LayerConstructor;

import javax.annotation.processing.*;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.TypeElement;
import javax.tools.Diagnostic;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Date;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@AutoService(Processor.class)
@SupportedSourceVersion(SourceVersion.RELEASE_8)
@SupportedAnnotationTypes("in.mcxiv.annotations.LayerConstructor")
public class VPProcessor extends AbstractProcessor {

    private Messager messager;

    @Override
    public synchronized void init(ProcessingEnvironment processingEnv) {
        super.init(processingEnv);
        messager = processingEnv.getMessager();
        println("VPProcessor initialized!");
    }

    @Override
    public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {

        Set<ExecutableElement> constructors = roundEnv.getElementsAnnotatedWith(LayerConstructor.class)
                .stream()
                .filter(element -> element.getKind() == ElementKind.CONSTRUCTOR)
                .map(element -> ((ExecutableElement) element))
                .collect(Collectors.toSet());

        TypeSpec.Builder layerVPLTypeSpecBuilder = TypeSpec
                .classBuilder("LayerVPL")
                .addModifiers(Modifier.PUBLIC)
                .superclass(ClassName.get("in.mcxiv.ai.convnet.net", "VPL"));

        for (ExecutableElement constructorElement : constructors) {

            String layerSubclassID = ((TypeElement) constructorElement.getEnclosingElement()).getQualifiedName().toString();
            layerSubclassID = layerSubclassID
                    .replaceAll("[lL]ayer$", "");
            String layerSubclassName = layerSubclassID.substring(layerSubclassID.lastIndexOf(".") + 1);

            String vpClassName = layerSubclassName + "VP";

            LayerConstructor layerConstructor = constructorElement.getAnnotation(LayerConstructor.class);
            String layerTag = layerConstructor.tag();
            String[] requiredFields = splitIntoAppropriateArray(layerConstructor.required());
            String[] optionalFields = splitIntoAppropriateArray(layerConstructor.optional());

            MethodSpec constructorSpec = createConstructor(requiredFields, layerTag);

            MethodSpec[] requiredSetters = createRequiredSetters(requiredFields, vpClassName);
            MethodSpec[] requiredGetters = createRequiredGetters(requiredFields);

            MethodSpec[] optionalSetters = createOptionalSetters(optionalFields, vpClassName);
            MethodSpec[] optionalGetters = createOptionalGetters(optionalFields);

            MethodSpec[] defaultSetters = createDefaultSetters(vpClassName);

            MethodSpec[] methods = concat(requiredSetters, requiredGetters, optionalSetters, optionalGetters, defaultSetters);

            TypeSpec typeSpec = createType(vpClassName, constructorSpec, methods);

            JavaFile javaFile = createJavaFile(typeSpec);

            writeJavaFile(javaFile);

            layerVPLTypeSpecBuilder
                    .addMethod(createVPCreatorMethod(requiredFields, layerTag, vpClassName));

        }

        JavaFile javaFile = createJavaFile(layerVPLTypeSpecBuilder.build());
        writeJavaFile(javaFile);

        return false;
    }

    private MethodSpec createConstructor(String[] args, String layerTag) {
        assert args.length % 2 == 0 : "Illegal Set! The number of arguments should be Even.";

        MethodSpec.Builder builder = MethodSpec.constructorBuilder()
                .addModifiers(Modifier.PUBLIC);

        if (args.length > 1) {
            StringBuilder paramList = new StringBuilder("super(");

            for (int i = 0; i < args.length; i += 2) {

                Class<?> fieldType = getType(args[i]);
                String fieldName = args[i + 1];

                builder.addParameter(fieldType, fieldName);
                paramList.append('"').append(fieldName).append('"').append(", ").append(fieldName).append(", ");

            }

            String params = paramList.substring(0, paramList.length() - 2) + ")";

            builder.addStatement(params);
        }
        builder.addStatement("add(\"type\", $S)", layerTag);

        return builder.build();
    }

    private MethodSpec createVPCreatorMethod(String[] args, String layerTag, String vpClassName) {
        assert args.length % 2 == 0 : "Illegal Set! The number of arguments should be Even.";

        MethodSpec.Builder methodBuilder = MethodSpec.methodBuilder(layerTag)
                .addModifiers(Modifier.PUBLIC)
                .returns(ClassName.get("in.mcxiv.gen", vpClassName));

        StringBuilder paramList = new StringBuilder(vpClassName).append(" vp = new ").append(vpClassName).append("(");
        if (args.length > 1) {

            for (int i = 0; i < args.length; i += 2) {

                Class<?> fieldType = getType(args[i]);
                String fieldName = args[i + 1];

                methodBuilder.addParameter(fieldType, fieldName);
                paramList.append(fieldName).append(", ");

            }

            String params = paramList.substring(0, paramList.length() - 2) + ")";
            methodBuilder.addStatement(params);

        } else methodBuilder.addStatement(paramList.append(")").toString());

        methodBuilder.addStatement("add(vp)");
        methodBuilder.addStatement("return vp");

        return methodBuilder.build();
    }

    private MethodSpec[] createRequiredSetters(String[] args, String vpClassName) {
        if (args.length <= 1) return new MethodSpec[0];

        MethodSpec[] specs = new MethodSpec[args.length / 2];

        for (int i = 0; i < args.length; i += 2) {

            Class<?> fieldType = getType(args[i]);
            String fieldName = args[i + 1];

            MethodSpec.Builder builder = MethodSpec.methodBuilder(fieldName)
                    .addModifiers(Modifier.PUBLIC)
                    .returns(ClassName.get("in.mcxiv.gen", vpClassName))
                    .addParameter(fieldType, fieldName)
                    .addStatement("add($S, $L)", fieldName, fieldName)
                    .addStatement("return this");

            specs[i / 2] = builder.build();
        }

        return specs;
    }

    private MethodSpec[] createRequiredGetters(String[] args) {
        if (args.length <= 1) return new MethodSpec[0];

        MethodSpec[] specs = new MethodSpec[args.length / 2];

        for (int i = 0; i < args.length; i += 2) {

            String fieldTypeName = args[i];
            Class<?> fieldType = getType(fieldTypeName);
            String fieldName = args[i + 1];

            MethodSpec.Builder builder = MethodSpec.methodBuilder(fieldName)
                    .addModifiers(Modifier.PUBLIC)
                    .returns(fieldType)
                    .addStatement("$L $L = $L($S)", fieldTypeName, fieldName, getMethodName(fieldTypeName), fieldName)
                    .addStatement("return $L", fieldName);

            specs[i / 2] = builder.build();
        }

        return specs;
    }

    private MethodSpec[] createOptionalSetters(String[] args, String vpClassName) {
        if (args.length <= 2) return new MethodSpec[0];
        String[] new_args = new String[(args.length / 3) * 2];
        for (int i = 0, k = 0; i < args.length; i += 3, k += 2) {
            new_args[k] = args[i];
            new_args[k + 1] = args[i + 1];
        }
        return createRequiredSetters(new_args, vpClassName);
    }

    private MethodSpec[] createOptionalGetters(String[] args) {
        if (args.length <= 2) return new MethodSpec[0];

        MethodSpec[] specs = new MethodSpec[args.length / 3];

        for (int i = 0; i < args.length; i += 3) {

            String fieldTypeName = args[i];
            Class<?> fieldType = getType(fieldTypeName);
            String fieldName = args[i + 1];
            String fieldDefault = args[i + 2];

            MethodSpec.Builder builder = MethodSpec.methodBuilder(fieldName)
                    .addModifiers(Modifier.PUBLIC)
                    .returns(fieldType)
                    .addStatement("$L $L = $L($S, $L)", fieldTypeName, fieldName, getMethodName(fieldTypeName), fieldName, fieldDefault)
                    .addStatement("return $L", fieldName);

            specs[i / 3] = builder.build();
        }

        return specs;
    }

    private MethodSpec[] createDefaultSetters(String vpClassName) {
        MethodSpec[] specs = new MethodSpec[1];

        String fieldName = "activation";

        MethodSpec.Builder builder = MethodSpec.methodBuilder(fieldName)
                .addModifiers(Modifier.PUBLIC)
                .returns(ClassName.get("in.mcxiv.gen", vpClassName))
                .addParameter(String.class, fieldName)
                .addStatement("add($S, $L)", fieldName, fieldName)
                .addStatement("return this");

        specs[0] = builder.build();

        return specs;
    }

    private TypeSpec createType(String vpClassName, MethodSpec constructorSpec, MethodSpec[] methods) {
        TypeSpec.Builder builder = TypeSpec
                .classBuilder(vpClassName)
                .addModifiers(Modifier.PUBLIC)
                .superclass(ClassName.get("in.mcxiv.ai.convnet.net", "VP"))
                .addMethod(constructorSpec);

        for (MethodSpec method : methods) builder.addMethod(method);

        return builder.build();
    }

    private JavaFile createJavaFile(TypeSpec typeSpec) {
        return JavaFile
                .builder("in.mcxiv.gen", typeSpec)
                .indent("    ")
                .addStaticImport(Date.class, "UTC")
                .addStaticImport(ClassName.get("java.time", "ZonedDateTime"), "*")
                .build();
    }

    private void writeJavaFile(JavaFile javaFile) {
        try {

            javaFile.writeTo(processingEnv.getFiler());

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void println(Object object) {
        messager.printMessage(Diagnostic.Kind.NOTE, object.toString());
    }

    private String[] splitIntoAppropriateArray(String args) {
        return args
                .replace(",", "")
                .replace("  ", "")
                .split(" ");
    }

    private String getMethodName(String arg) {
        switch (arg) {
            case "int":
                return "getInt";
            case "double":
                return "getD";
            case "String":
                return "getSt";
            default:
                throw new AssertionError("No predefined class object defined for type " + arg);
        }
    }

    private Class<?> getType(String arg) {
        switch (arg) {
            case "int":
                return Integer.TYPE;
            case "double":
                return Double.TYPE;
            case "String":
                return String.class;
            default:
                throw new AssertionError("No predefined class object defined for type " + arg);
        }
    }

    @SafeVarargs
    private static <T> T[] concat(T[]... a) {
        T[] t = concat(a[0], a[1]);
        for (int i = 2; i < a.length; i++) {
            t = concat(t, a[i]);
        }
        return t;
    }

    private static <T> T[] concat(T[] a1, T[] a2) {
        return Stream.concat(Arrays.stream(a1), Arrays.stream(a2))
                .toArray(size -> (T[]) Array.newInstance(a1.getClass().getComponentType(), size));
    }

}
