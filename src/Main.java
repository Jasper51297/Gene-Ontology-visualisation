/**
 * This java class contains the graphical user interface for the GO visualizer
 *
 * Versionnumbers:
 * Java jdk 1.8.0_172
 * Javafx 8.0172-b11
 */

//Imports
import javafx.application.Application;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.event.ActionEvent;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;

public class Main extends Application {
    private VBox vb3 = new VBox();
    private HBox hb5 = new HBox();
    private HBox hb6 = new HBox();
    private ToggleGroup group = new ToggleGroup();
    private Boolean select1 = false;
    private Boolean select2 = false;
    private Boolean select3 = false;
    private Button button4 = new Button("Visualize");
    private String obofile = "";
    private String mod1 = "";
    private String mod2 = "";
    private String importance = "";
    private TextField tf2 = new TextField();
    private TextField tf3 = new TextField();
    private TextField tf4 = new TextField();
    private CheckBox cb1 = new CheckBox();

    /**
     * This method fills a HBox with widgets that are necessary for selecting
     * the OBO file. The method enables the "Visualize" button if all the necessary
     * files are submitted.
     *
     * @return HBox with all widgets that are necessary for selecting the OBO file.
     */
    private HBox horizontaal(){
        Label label7 = new Label("OBO file           ");
        label7.setFont(new Font("Arial", 15));
        HBox hb1 = new HBox();
        hb1.setPadding(new Insets(15, 0, 0, 0));
        TextField tf = new TextField();
        tf.setDisable(true);
        tf.setPromptText("OBO file");
        tf.setPrefWidth(200);
        tf.setFocusTraversable(false);
        Button button1 = new Button("Select file");
        button1.setOnAction(e -> {
            FileChooser fileChooser = new FileChooser();
            FileChooser.ExtensionFilter extfilter = new FileChooser.ExtensionFilter("OBO files (*.obo)", "*.obo");
            FileChooser.ExtensionFilter extfilter1 = new FileChooser.ExtensionFilter("All files (*.*)", "*.*");
            fileChooser.getExtensionFilters().addAll(extfilter, extfilter1);
            File selectedFile = fileChooser.showOpenDialog(null);
            if (selectedFile != null){
                obofile = selectedFile.getAbsolutePath();
                tf.setText(selectedFile.getName());
                select1 = true;
                if(select1 && select2 && select3 && group.getSelectedToggle().getUserData().equals("2")){
                    button4.setDisable(false);
                }
                if(select1 && select2 && group.getSelectedToggle().getUserData().equals("1")){
                    button4.setDisable(false);
                }
            } else {
                System.out.println("file is not valid");
            }
        });

        hb1.getChildren().addAll(label7, tf,button1);
        hb1.setAlignment(Pos.CENTER);
        return hb1;
    }

    /**
     * This method creates a HBox that contains a radiobutton group
     * that controls if the input field for model 2 should be visible.
     * @return HBox that contains a radiobutton group.
     */
    private HBox horizontaal2() {
        HBox hb2 = new HBox();
        hb2.setSpacing(80);
        RadioButton rb1 = new RadioButton("1");
        rb1.setToggleGroup(group);
        rb1.setUserData("1");
        RadioButton rb2 = new RadioButton("2");
        rb2.setToggleGroup(group);
        rb2.setUserData("2");
        group.selectToggle(rb1);
        hb2.getChildren().addAll(rb1, rb2);
        hb2.setAlignment(Pos.CENTER);
        group.selectedToggleProperty().addListener(new ChangeListener<Toggle>() {
            @Override
            public void changed(ObservableValue<? extends Toggle> observable, Toggle oldValue, Toggle newValue) {
                if(group.getSelectedToggle().getUserData().equals("1")){
                    vb3.setVisible(false);
                    tf3.setText("");
                }
                else{
                    vb3.setVisible(true);
                    button4.setDisable(true);
                }
            }
        });
        return hb2;
    }

    /**
     * This method fills a VBox with widgets that are necessary for selecting
     * the model 1 file. The method enables the "Visualize" button if all the necessary
     * files are submitted.
     * @return A VBox with widgets that are necessary for selecting the model 1 file.
     */
    private VBox dubbel(){
        HBox hb4 = new HBox();
        tf2.setPromptText("model 1");
        tf2.setDisable(true);
        Button button2 = new Button("Select file");
        button2.setOnAction((ActionEvent e) -> {
            FileChooser fileChooser = new FileChooser();
            FileChooser.ExtensionFilter extfilter = new FileChooser.ExtensionFilter("TXT files (*.txt)", "*.txt");
            FileChooser.ExtensionFilter extfilter1 = new FileChooser.ExtensionFilter("All files (*.*)", "*.*");
            fileChooser.getExtensionFilters().addAll(extfilter, extfilter1);
            File selectedFile = fileChooser.showOpenDialog(null);
            if (selectedFile != null){
                mod1 = selectedFile.getAbsolutePath();
                tf2.setText(selectedFile.getName());
                select2 = true;
                if(select1 && select2 && select3 && group.getSelectedToggle().getUserData().equals("2")){
                    button4.setDisable(false);
                }
                if(select1 && select2 && group.getSelectedToggle().getUserData().equals("1")){
                    button4.setDisable(false);
                }
            } else {
                System.out.println("file is not valid");
            }
        });
        VBox vb1 = new VBox();
        Label label5 = new Label("Model 1 file");
        label5.setFont(new Font("Arial", 15));
        hb4.getChildren().addAll(tf2, button2);
        hb4.setAlignment(Pos.CENTER);
        vb1.setAlignment(Pos.CENTER);
        vb1.getChildren().addAll(label5, hb4);
        return vb1;
    }

    /**
     * This method fills a box with widgets that are necessary for selecting
     * the model 2 file. The method enables the "Visualize" button if all the necessary
     * files are submitted.
     */
    private void dubbel2(){
        tf3.setPromptText("model 2");
        tf3.setDisable(true);
        Button button3 = new Button("Select file");
        button3.setOnAction(e -> {
            FileChooser fileChooser = new FileChooser();
            FileChooser.ExtensionFilter extfilter = new FileChooser.ExtensionFilter("TXT files (*.txt)", "*.txt");
            FileChooser.ExtensionFilter extfilter1 = new FileChooser.ExtensionFilter("All files (*.*)", "*.*");
            fileChooser.getExtensionFilters().addAll(extfilter, extfilter1);
            File selectedFile = fileChooser.showOpenDialog(null);
            if (selectedFile != null){
                mod2 = selectedFile.getAbsolutePath();
                tf3.setText(selectedFile.getName());
                select3 = true;
                if(select1 && select2 && select3 && group.getSelectedToggle().getUserData().equals("2")){
                    button4.setDisable(false);
                }
                if(select1 && select2 && group.getSelectedToggle().getUserData().equals("1")){
                    button4.setDisable(false);
                }
            } else {
                System.out.println("file is not valid");
            }
        });
        Label label6 = new Label("Model 2 file");
        label6.setFont(new Font("Arial", 15));
        hb5.getChildren().addAll(tf3, button3);
        hb5.setAlignment(Pos.CENTER);
        vb3.setVisible(false);
        vb3.setAlignment(Pos.CENTER);
        vb3.getChildren().addAll(label6, hb5);
    }

    /**
     * This method fills a box with widgets that are necessary for selecting
     * the importance file.
     */
    private void dubbel3(){
        Label label8 = new Label("Importance file ");
        label8.setFont(new Font("Arial", 15));
        tf4.setPromptText("Importance file (Optional)");
        tf4.setDisable(true);
        tf4.setPrefWidth(200);
        Button button5 = new Button("Select file");
        button5.setOnAction(e -> {
            FileChooser fileChooser = new FileChooser();
            FileChooser.ExtensionFilter extfilter = new FileChooser.ExtensionFilter("TXT files (*.txt)", "*.txt");
            FileChooser.ExtensionFilter extfilter1 = new FileChooser.ExtensionFilter("All files (*.*)", "*.*");
            fileChooser.getExtensionFilters().addAll(extfilter, extfilter1);
            File selectedFile = fileChooser.showOpenDialog(null);
            if (selectedFile != null){
                importance = selectedFile.getAbsolutePath();
                tf4.setText(selectedFile.getName());
            } else {
                System.out.println("file is not valid");
            }
        });
        hb6.getChildren().addAll(label8, tf4, button5);
        hb6.setAlignment(Pos.CENTER);
        hb6.setVisible(true);
    }

    /**
     * This method fills a HBox with 2 VBoxes that contain widgets
     * that are necessary for selecting the model files.
     * @return A HBox with the widgets that are necessary for selecting the model files.
     */
    private HBox horizontaal3(){
        HBox hb3 = new HBox();
        hb3.setSpacing(50);
        VBox vb4 = dubbel();
        dubbel2();
        hb3.getChildren().addAll(vb4, vb3);
        hb3.setAlignment(Pos.CENTER);
        return hb3;
    }

    /**
     * This method puts the model widget box, cut first checkbox and the "Visualize" button in a VBox
     * @param button Visualize button which is added to the bottom of the VBox.
     * @return VBox with the model widget box, cut first checkbox and the "Visualize" button.
     */
    private VBox verticaal(Button button){
        VBox vb2 = new VBox();
        vb2.setPadding(new Insets(50, 0, 0, 0));
        vb2.setSpacing(50);
        Label label3 = new Label("Number of models");
        label3.setFont(new Font("Arial", 20));
        HBox hb2 = horizontaal2();
        HBox hb3 = horizontaal3();
        HBox hb4 = new HBox(20);
        Label label4 = new Label("Remove nodes without score");
        label4.setFont(new Font("Arial", 15));
        hb4.getChildren().addAll(label4, cb1);
        cb1.setSelected(true);
        hb4.setAlignment(Pos.CENTER);
        vb2.getChildren().addAll(label3, hb2, hb3, hb4, button);
        vb2.setAlignment(Pos.CENTER);
        return vb2;
    }

    /**
     * This method puts all the boxes in a VBox and adds an image to it.
     * It also writes the run parameters to a file when the "Visualize" button is clicked.
     * @param primaryStage Stage where the final VBox is displayed.
     * @return VBox with all the content that should be displayed
     */
    private VBox screen(Stage primaryStage){
        VBox vb1 = new VBox(5);
        vb1.setPadding(new Insets(15, 0, 0, 0));
        vb1.setSpacing(10);
        Label label1 = new Label("GO model visualizer");
        label1.setFont(new Font("Arial", 30));
        Label label2 = new Label("Version 1.0");
        label2.setFont(new Font("Arial", 15));
        final ImageView imv = new ImageView();
        final Image image2 = new Image(getClass().getResource("tudelft.jpg").toExternalForm());
        imv.setImage(image2);
        button4.setPrefSize(200,50);
        button4.setDisable(true);
        button4.setOnAction((ActionEvent e) -> {
            String outputArgs = obofile;
            String bokehArgs = "";
            if(group.getSelectedToggle().getUserData().equals("1")){
                outputArgs += " --model1 " + mod1;
                bokehArgs += "--models 1";
            }
            if(group.getSelectedToggle().getUserData().equals("2")){
                outputArgs += " --model1 " + mod1;
                outputArgs += " --model2 " + mod2;
                bokehArgs += " --models 2";
            }
            if(!tf4.getText().isEmpty()){
                outputArgs += " --importance " + importance;
                bokehArgs += " --importance";
            }
            if(cb1.isSelected()){
                outputArgs += " --cutfirst";
            }
            try (BufferedWriter bw = new BufferedWriter(new PrintWriter("runargs.txt"))) {
                bw.write(outputArgs);
            } catch (IOException error) {
                error.printStackTrace();
            }
            try (BufferedWriter bw = new BufferedWriter(new PrintWriter("bokehargs.txt"))) {
                bw.write(bokehArgs);
            } catch (IOException error) {
                error.printStackTrace();
            }
            System.exit(0);
        });
        HBox hb1 = horizontaal();
        VBox vb2 = verticaal(button4);
        dubbel3();
        vb1.getChildren().addAll(label1, label2, imv, hb1, hb6, vb2);
        vb1.setAlignment(Pos.TOP_CENTER);
        return vb1;
    }

    /**
     * Shows the application.
     * @param primaryStage Stage where the final VBox is displayed.
     * @throws Exception Throwed when something goes wrong.
     */
    @Override
    public void start(Stage primaryStage) throws Exception{
        VBox mvb1 = screen(primaryStage);
        Scene scene = new Scene(mvb1, 800, 800, Color.WHITE);
        primaryStage.setTitle("GO model visualizer 1.0");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    /**
     * Starts the application
     * @param args Adds args to the launch method if supplied.
     */
    public static void main(String[] args) {
        launch(args);
    }
}