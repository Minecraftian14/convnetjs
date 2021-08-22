package in.mcxiv.proc;

import com.squareup.javapoet.JavaFile;
import in.mcxiv.annotations.VPConstructor;

import javax.annotation.processing.Messager;
import javax.annotation.processing.ProcessingEnvironment;
import javax.annotation.processing.RoundEnvironment;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.ExecutableElement;
import javax.tools.Diagnostic;
import java.io.IOException;
import java.util.Set;
import java.util.stream.Collectors;

public class ProcessorUtilities {

    private final ProcessingEnvironment processingEnv;
    private final Messager messager;

    public ProcessorUtilities(ProcessingEnvironment processingEnv, Messager messager) {
        this.processingEnv = processingEnv;
        this.messager = messager;
    }

    static Set<ExecutableElement> getConstructors(RoundEnvironment roundEnv) {
        return roundEnv.getElementsAnnotatedWith(VPConstructor.class)
                .stream()
                .filter(element -> element.getKind() == ElementKind.CONSTRUCTOR)
                .map(element -> ((ExecutableElement) element))
                .collect(Collectors.toSet());
    }

    void writeJavaFile(JavaFile javaFile) {
        try {

            javaFile.writeTo(processingEnv.getFiler());

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    void println(Object object) {
        messager.printMessage(Diagnostic.Kind.NOTE, object.toString());
    }

}
