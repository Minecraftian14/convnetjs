package in.mcxiv.ai.convnet;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;

import java.io.BufferedReader;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public enum Dataset {
    IRIS("Iris.csv");

    public static int trainingSamples = 1000;

    public Vol[] xi;
    public int[] yi;
    private final String name;

    Dataset(String name) {
        this.name = name;
    }

    public void load() {
        try {

            URL resource = Dataset.class.getClassLoader().getResource(name);
            assert resource != null;
            BufferedReader reader = Files.newBufferedReader(Paths.get(resource.toURI()));
            CSVReader csvReader = new CSVReader(reader);

            csvReader.readNext();//skip the first row
            List<Vol> xList = new ArrayList<>();
            List<Integer> yList = new ArrayList<>();

            String[] line;
            for (int i = 0; i < trainingSamples && (line = csvReader.readNext()) != null; i++) {

                Vol x = new Vol(line.length - 1, 1, 1);

                for (int xidx = 0; xidx < line.length - 1; xidx++) {
                    x.set(xidx, 0, 0, Double.parseDouble(line[xidx]));
                }

                xList.add(x);
                yList.add(Integer.parseInt(line[line.length - 1]));
            }

            reader.close();
            csvReader.close();

            xi = xList.toArray(Vol[]::new);
            yi = new int[yList.size()];
            for (int i = 0, s = yList.size(); i < s; i++) yi[i] = yList.get(i);

        } catch (IOException | CsvValidationException | URISyntaxException e) {
            e.printStackTrace();
        }
    }

}
