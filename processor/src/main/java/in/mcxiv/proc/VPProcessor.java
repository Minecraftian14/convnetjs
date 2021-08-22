package in.mcxiv.proc;

import com.google.auto.service.AutoService;
import com.squareup.javapoet.JavaFile;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.TypeSpec;
import in.mcxiv.annotations.VPConstructor;

import javax.annotation.processing.*;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.TypeElement;
import java.util.Set;

import static in.mcxiv.proc.JPoetUtilities.*;

@AutoService(Processor.class)
@SupportedSourceVersion(SourceVersion.RELEASE_8)
@SupportedAnnotationTypes("in.mcxiv.annotations.VPConstructor")
public class VPProcessor extends AbstractProcessor {

    private ProcessorUtilities utilities;

    @Override
    public synchronized void init(ProcessingEnvironment processingEnv) {
        super.init(processingEnv);
        utilities = new ProcessorUtilities(processingEnv, processingEnv.getMessager());
        utilities.println("VPProcessor initialized!");
    }

    @Override
    public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {

        Set<ExecutableElement> constructors = ProcessorUtilities.getConstructors(roundEnv);

        utilities.println("<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>");
        for (ExecutableElement constructor : constructors) {
            utilities.println(constructor);
        }

        if (constructors.isEmpty()) return false;

        TypeSpec.Builder layerVPLTypeSpecBuilder = createVPSubType("LayerVPL", "VPL", null, null);

        for (ExecutableElement constructorElement : constructors) {

            String layerSubclassID = ((TypeElement) constructorElement.getEnclosingElement()).getQualifiedName().toString();
            layerSubclassID = layerSubclassID
                    .replaceAll("[lL]ayer$", "");
            String layerSubclassName = layerSubclassID.substring(layerSubclassID.lastIndexOf(".") + 1);

            String vpClassName = layerSubclassName + "VP";

            VPConstructor VPConstructor = constructorElement.getAnnotation(VPConstructor.class);
            String layerTag = VPConstructor.tag();
            String[] requiredFields = BasicUtilities.splitIntoAppropriateArray(VPConstructor.required());
            String[] optionalFields = BasicUtilities.splitIntoAppropriateArray(VPConstructor.optional());

            MethodSpec constructorSpec = createVPConstructor(requiredFields, layerTag);

            MethodSpec[] requiredSetters = createRequiredSetters(requiredFields, vpClassName);
            MethodSpec[] requiredGetters = createRequiredGetters(requiredFields);

            MethodSpec[] optionalSetters = createOptionalSetters(optionalFields, vpClassName);
            MethodSpec[] optionalGetters = createOptionalGetters(optionalFields);

            MethodSpec[] defaultSetters = createDefaultSetters(vpClassName);

            MethodSpec[] methods = BasicUtilities.concat(requiredSetters, requiredGetters, optionalSetters, optionalGetters, defaultSetters);

            TypeSpec typeSpec = createVPSubType(vpClassName, "VP", constructorSpec, methods).build();

            JavaFile javaFile = createJavaFile(typeSpec);

            utilities.writeJavaFile(javaFile);

            if (!layerTag.equals("trainer"))
                layerVPLTypeSpecBuilder
                        .addMethod(createVPCreatorMethod(requiredFields, layerTag, vpClassName));

        }

        JavaFile javaFile = createJavaFile(layerVPLTypeSpecBuilder.build());
        utilities.writeJavaFile(javaFile);

        utilities.println("<<<<<<<<<<<<<<<<<<<<<Processed >>>>>>>>>>>>>>>>>>>>>");
        return true;
    }


}
