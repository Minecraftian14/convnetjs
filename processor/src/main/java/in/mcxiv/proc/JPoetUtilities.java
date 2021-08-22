package in.mcxiv.proc;

import com.squareup.javapoet.ClassName;
import com.squareup.javapoet.JavaFile;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.TypeSpec;

import javax.lang.model.element.Modifier;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import static in.mcxiv.proc.BasicUtilities.onlyThird;
import static in.mcxiv.proc.BasicUtilities.removeEveryThird;

public class JPoetUtilities {

    @FunctionalInterface
    private interface FieldConsumer {
        void consume(String fieldTypeName, Class<?> fieldType, String fieldName);
    }

    @FunctionalInterface
    private interface IndexedFieldConsumer {
        void consume(int index, String fieldTypeName, Class<?> fieldType, String fieldName);
    }

    private static void forEachField(String[] args, IndexedFieldConsumer consumer) {
        if (args.length <= 1) return;
        for (int i = 0, k=0; i < args.length; i += 2, k++) {
            String fieldTypeName = args[i];
            Class<?> fieldType = VPUtilities.getType(fieldTypeName);
            String fieldName = args[i + 1];
            consumer.consume(k, fieldTypeName, fieldType, fieldName);
        }
    }

    private static void forEachField(String[] args, FieldConsumer consumer) {
        forEachField(args, (index, fieldTypeName, fieldType, fieldName) -> consumer.consume(fieldTypeName, fieldType, fieldName));
    }

    private static MethodSpec.Builder createPublicConstructor() {
        return MethodSpec
                .constructorBuilder()
                .addModifiers(Modifier.PUBLIC);
    }

    static TypeSpec.Builder createVPSubType(String className, String superClass, MethodSpec constructorSpec, MethodSpec[] methods) {
        TypeSpec.Builder builder = TypeSpec
                .classBuilder(className)
                .addModifiers(Modifier.PUBLIC)
                .superclass(ClassName.get("in.mcxiv.ai.convnet.net", superClass));

        if (constructorSpec != null)
            builder.addMethod(constructorSpec);

        if (methods != null)
            for (MethodSpec method : methods) builder.addMethod(method);

        return builder;
    }

    private static MethodSpec.Builder createPublicMethod(String name) {
        return MethodSpec
                .methodBuilder(name)
                .addModifiers(Modifier.PUBLIC);
    }

    private static void createParameters(String[] args, MethodSpec.Builder methodBuilder) {
        if (args.length <= 1) return;
        forEachField(args, (ftn, ft, fn) -> methodBuilder.addParameter(ft, fn));
    }

    private static String createParameterCall(String[] args) {
        if (args.length <= 1) return "";
        StringBuilder builder = new StringBuilder();
        forEachField(args, (ftn, ft, fn) -> builder.append(fn).append(", "));
        return builder.substring(0, builder.length() - 2);
    }

    private static String createQuotedParameterCall(String[] args) {
        if (args.length <= 1) return "";
        StringBuilder builder = new StringBuilder();
        forEachField(args, (ftn, ft, fn) ->
                builder.append('"').append(fn).append('"').append(", ").append(fn).append(", "));
        return builder.substring(0, builder.length() - 2);
    }

    static MethodSpec createVPConstructor(String[] args, String layerTag) {
        assert args.length % 2 == 0 : "Illegal Set! The number of arguments should be Even.";

        MethodSpec.Builder builder = createPublicConstructor();
        createParameters(args, builder);

        builder.addStatement(String.format("super(%s)", createQuotedParameterCall(args)));
        builder.addStatement("add(\"type\", $S)", layerTag);

        return builder.build();
    }

    static MethodSpec createVPCreatorMethod(String[] args, String layerTag, String vpClassName) {
        assert args.length % 2 == 0 : "Illegal Set! The number of arguments should be Even.";

        MethodSpec.Builder methodBuilder = createPublicMethod(layerTag)
                .returns(ClassName.get("in.mcxiv.gen", vpClassName));
        createParameters(args, methodBuilder);

        methodBuilder.addStatement(String.format("%s vp = new %s(%s)", vpClassName, vpClassName, createParameterCall(args)));
        methodBuilder.addStatement("add(vp)");
        methodBuilder.addStatement("return vp");

        return methodBuilder.build();
    }

    private static MethodSpec[] createSetters(String[] args, String vpClassName) {
        if (args.length <= 1) return new MethodSpec[0];

        List<MethodSpec> specs = new ArrayList<>();
        forEachField(args, (ftn, ft, fn) -> specs.add(createPublicMethod(fn)
                .returns(ClassName.get("in.mcxiv.gen", vpClassName))
                .addParameter(ft, fn)
                .addStatement("add($S, $L)", fn, fn)
                .addStatement("return this")
                .build()));

        return specs.toArray(new MethodSpec[0]);
    }

    private static MethodSpec[] createGetters(String[] args) {
        if (args.length <= 1) return new MethodSpec[0];

        List<MethodSpec> specs = new ArrayList<>();
        forEachField(args, (ftn, ft, fn) -> specs.add(createPublicMethod(fn)
                .returns(ft)
                .addStatement("$L $L = $L($S)", ftn, fn, VPUtilities.getMethodName(ftn), fn)
                .addStatement("return $L", fn)
                .build()));

//                    .returns(fieldType)
//                .addStatement("$L $L = $L($S, $L)", fieldTypeName, fieldName, VPUtilities.getMethodName(fieldTypeName), fieldName, fieldDefault)
//                .addStatement("return $L", fieldName);

        return specs.toArray(new MethodSpec[0]);
    }

    private static MethodSpec[] createGetters(String[] args, String[] defaultValues) {
        if (args.length <= 1) return new MethodSpec[0];

        List<MethodSpec> specs = new ArrayList<>();

        forEachField(args, (i, ftn, ft, fn) -> specs.add(createPublicMethod(fn)
                .returns(ft)
                .addStatement("$L $L = $L($S, $L)", ftn, fn, VPUtilities.getMethodName(ftn), fn, defaultValues[i])
                .addStatement("return $L", fn)
                .build()));

        return specs.toArray(new MethodSpec[0]);
    }

    static MethodSpec[] createRequiredSetters(String[] args, String vpClassName) {
        return createSetters(args, vpClassName);
    }

    static MethodSpec[] createRequiredGetters(String[] args) {
        return createGetters(args);
    }

    static MethodSpec[] createOptionalSetters(String[] args, String vpClassName) {
        return createSetters(removeEveryThird(args), vpClassName);
    }

    static MethodSpec[] createOptionalGetters(String[] args) {
        return createGetters(removeEveryThird(args), onlyThird(args));
    }

    static MethodSpec[] createDefaultSetters(String vpClassName) {
        return createSetters(new String[]{"String", "activation"}, vpClassName);
    }

    static JavaFile createJavaFile(TypeSpec typeSpec) {
        return JavaFile
                .builder("in.mcxiv.gen", typeSpec)
                .indent("    ")
                .addStaticImport(Date.class, "UTC")
                .addStaticImport(ClassName.get("java.time", "ZonedDateTime"), "*")
                .build();
    }

}
