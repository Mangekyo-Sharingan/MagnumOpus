# MagnumOpus Project Flowchart

This document contains Mermaid flowcharts that visualize the architecture and workflow of the Diabetic Retinopathy Classification project.

## 🏗️ Overall Project Architecture

```mermaid
graph TB
    %% Data Sources
    APTOS[(APTOS 2019<br/>Dataset)]
    EYEPACS[(EyePACS<br/>Dataset)]
    
    %% Configuration
    CONFIG[Config Module<br/>config.py]
    
    %% Data Pipeline
    DATA_LOADER[CustomDataLoader<br/>data.py]
    PIPELINE_A[Pipeline A<br/>224x224<br/>VGG16/ResNet50]
    PIPELINE_B[Pipeline B<br/>299x299<br/>InceptionV3]
    
    %% Models
    VGG16[VGG16 Model]
    RESNET[ResNet50 Model]
    INCEPTION[InceptionV3 Model]
    
    %% Training
    TRAINER[Trainer<br/>train.py]
    
    %% Evaluation
    EVALUATOR[Evaluator<br/>test.py]
    
    %% Utilities
    UTILS[Utils<br/>utils.py]
    
    %% Main Entry
    MAIN[main.py<br/>Entry Point]
    TESTS[run_module_tests.py<br/>Testing System]
    
    %% Connections
    APTOS --> DATA_LOADER
    EYEPACS --> DATA_LOADER
    CONFIG --> DATA_LOADER
    CONFIG --> TRAINER
    CONFIG --> EVALUATOR
    CONFIG --> VGG16
    CONFIG --> RESNET
    CONFIG --> INCEPTION
    
    DATA_LOADER --> PIPELINE_A
    DATA_LOADER --> PIPELINE_B
    
    PIPELINE_A --> VGG16
    PIPELINE_A --> RESNET
    PIPELINE_B --> INCEPTION
    
    VGG16 --> TRAINER
    RESNET --> TRAINER
    INCEPTION --> TRAINER
    
    TRAINER --> EVALUATOR
    
    UTILS -.-> DATA_LOADER
    UTILS -.-> TRAINER
    UTILS -.-> EVALUATOR
    
    MAIN --> CONFIG
    MAIN --> DATA_LOADER
    MAIN --> TRAINER
    MAIN --> EVALUATOR
    
    TESTS -.-> CONFIG
    TESTS -.-> DATA_LOADER
    TESTS -.-> VGG16
    TESTS -.-> RESNET
    TESTS -.-> INCEPTION
    TESTS -.-> TRAINER
    TESTS -.-> EVALUATOR
    TESTS -.-> UTILS
    
    %% Styling
    classDef dataSource fill:#e1f5fe
    classDef module fill:#f3e5f5
    classDef model fill:#e8f5e8
    classDef pipeline fill:#fff3e0
    classDef main fill:#ffebee
    
    class APTOS,EYEPACS dataSource
    class CONFIG,DATA_LOADER,TRAINER,EVALUATOR,UTILS module
    class VGG16,RESNET,INCEPTION model
    class PIPELINE_A,PIPELINE_B pipeline
    class MAIN,TESTS main
```

## 🔄 Data Processing Workflow

```mermaid
flowchart TD
    START([Start Data Loading])
    
    %% Configuration
    LOAD_CONFIG[Load Configuration]
    
    %% Data Loading
    CHECK_APTOS{APTOS CSV<br/>exists?}
    LOAD_APTOS[Load APTOS<br/>train.csv & test.csv]
    VERIFY_APTOS[Verify APTOS<br/>image files]
    
    CHECK_EYEPACS{EyePACS CSV<br/>exists?}
    LOAD_EYEPACS[Load EyePACS<br/>trainLabels.csv]
    STANDARDIZE[Standardize column names<br/>image → id_code<br/>level → diagnosis]
    VERIFY_EYEPACS[Verify EyePACS<br/>image files]
    
    %% Merging
    MERGE[Merge Datasets]
    SHUFFLE[Shuffle merged data<br/>with random_state=42]
    STATS[Print dataset statistics<br/>- Total samples<br/>- Dataset distribution<br/>- Diagnosis distribution]
    
    %% Pipeline Selection
    SPLIT[Train/Validation Split<br/>80/20 stratified]
    SELECT_PIPELINE{Model Type?}
    
    %% Pipelines
    PIPELINE_A_PROC[Pipeline A Processing<br/>- Resize to 224x224<br/>- Data augmentation<br/>- Normalization]
    PIPELINE_B_PROC[Pipeline B Processing<br/>- Resize to 299x299<br/>- Data augmentation<br/>- Normalization]
    
    %% DataLoaders
    CREATE_LOADERS[Create PyTorch<br/>DataLoaders]
    
    COMPLETE([Data Loading Complete])
    
    %% Flow
    START --> LOAD_CONFIG
    LOAD_CONFIG --> CHECK_APTOS
    
    CHECK_APTOS -->|Yes| LOAD_APTOS
    CHECK_APTOS -->|No| CHECK_EYEPACS
    LOAD_APTOS --> VERIFY_APTOS
    VERIFY_APTOS --> CHECK_EYEPACS
    
    CHECK_EYEPACS -->|Yes| LOAD_EYEPACS
    CHECK_EYEPACS -->|No| MERGE
    LOAD_EYEPACS --> STANDARDIZE
    STANDARDIZE --> VERIFY_EYEPACS
    VERIFY_EYEPACS --> MERGE
    
    MERGE --> SHUFFLE
    SHUFFLE --> STATS
    STATS --> SPLIT
    SPLIT --> SELECT_PIPELINE
    
    SELECT_PIPELINE -->|VGG16/ResNet50| PIPELINE_A_PROC
    SELECT_PIPELINE -->|InceptionV3| PIPELINE_B_PROC
    
    PIPELINE_A_PROC --> CREATE_LOADERS
    PIPELINE_B_PROC --> CREATE_LOADERS
    CREATE_LOADERS --> COMPLETE
    
    %% Styling
    classDef startEnd fill:#c8e6c9
    classDef process fill:#bbdefb
    classDef decision fill:#ffe0b2
    classDef pipeline fill:#f8bbd9
    
    class START,COMPLETE startEnd
    class LOAD_CONFIG,LOAD_APTOS,VERIFY_APTOS,LOAD_EYEPACS,STANDARDIZE,VERIFY_EYEPACS,MERGE,SHUFFLE,STATS,SPLIT,CREATE_LOADERS process
    class CHECK_APTOS,CHECK_EYEPACS,SELECT_PIPELINE decision
    class PIPELINE_A_PROC,PIPELINE_B_PROC pipeline
```

## 🎯 Training Workflow

```mermaid
flowchart TD
    START_TRAIN([Start Training])
    
    %% Setup
    SETUP_COMPONENTS[Setup Training Components<br/>- Adam Optimizer<br/>- CrossEntropyLoss<br/>- ReduceLROnPlateau]
    
    %% Training Loop
    EPOCH_START[Start Epoch]
    TRAIN_PHASE[Training Phase<br/>- Forward pass<br/>- Loss calculation<br/>- Backpropagation<br/>- Optimizer step]
    
    VAL_PHASE[Validation Phase<br/>- Forward pass<br/>- Loss calculation<br/>- No gradients]
    
    UPDATE_SCHEDULER[Update Learning Rate<br/>Scheduler]
    
    CHECK_BEST{Best validation<br/>loss?}
    SAVE_BEST[Save best model state]
    
    RECORD_HISTORY[Record training history<br/>- Train/Val loss<br/>- Train/Val accuracy]
    
    CHECK_EPOCHS{More epochs<br/>remaining?}
    
    LOAD_BEST[Load best model state]
    TRAINING_COMPLETE([Training Complete])
    
    %% Flow
    START_TRAIN --> SETUP_COMPONENTS
    SETUP_COMPONENTS --> EPOCH_START
    EPOCH_START --> TRAIN_PHASE
    TRAIN_PHASE --> VAL_PHASE
    VAL_PHASE --> UPDATE_SCHEDULER
    UPDATE_SCHEDULER --> CHECK_BEST
    
    CHECK_BEST -->|Yes| SAVE_BEST
    CHECK_BEST -->|No| RECORD_HISTORY
    SAVE_BEST --> RECORD_HISTORY
    
    RECORD_HISTORY --> CHECK_EPOCHS
    CHECK_EPOCHS -->|Yes| EPOCH_START
    CHECK_EPOCHS -->|No| LOAD_BEST
    LOAD_BEST --> TRAINING_COMPLETE
    
    %% Styling
    classDef startEnd fill:#c8e6c9
    classDef process fill:#bbdefb
    classDef decision fill:#ffe0b2
    classDef important fill:#ffcdd2
    
    class START_TRAIN,TRAINING_COMPLETE startEnd
    class SETUP_COMPONENTS,TRAIN_PHASE,VAL_PHASE,UPDATE_SCHEDULER,RECORD_HISTORY,LOAD_BEST process
    class CHECK_BEST,CHECK_EPOCHS decision
    class SAVE_BEST important
```

## 🧪 Testing & Evaluation Workflow

```mermaid
flowchart TD
    START_EVAL([Start Evaluation])
    
    %% Model Evaluation
    LOAD_MODEL[Load trained model]
    SET_EVAL_MODE[Set model to eval mode]
    
    PROCESS_BATCH[Process test batch<br/>- Forward pass<br/>- Get predictions<br/>- Get probabilities]
    
    MORE_BATCHES{More batches?}
    
    %% Metrics Calculation
    CALC_BASIC[Calculate basic metrics<br/>- Accuracy<br/>- Precision, Recall, F1]
    
    CALC_WEIGHTED[Calculate weighted averages]
    
    CLASSIFICATION_REPORT[Generate classification report]
    
    %% Visualization
    PLOT_CONFUSION[Plot confusion matrix]
    PLOT_DISTRIBUTION[Plot class distribution]
    
    %% Analysis
    FIND_MISCLASSIFIED[Find misclassified samples<br/>with highest confidence]
    
    SAVE_RESULTS[Save results to JSON]
    
    EVALUATION_COMPLETE([Evaluation Complete])
    
    %% Flow
    START_EVAL --> LOAD_MODEL
    LOAD_MODEL --> SET_EVAL_MODE
    SET_EVAL_MODE --> PROCESS_BATCH
    PROCESS_BATCH --> MORE_BATCHES
    
    MORE_BATCHES -->|Yes| PROCESS_BATCH
    MORE_BATCHES -->|No| CALC_BASIC
    
    CALC_BASIC --> CALC_WEIGHTED
    CALC_WEIGHTED --> CLASSIFICATION_REPORT
    CLASSIFICATION_REPORT --> PLOT_CONFUSION
    PLOT_CONFUSION --> PLOT_DISTRIBUTION
    PLOT_DISTRIBUTION --> FIND_MISCLASSIFIED
    FIND_MISCLASSIFIED --> SAVE_RESULTS
    SAVE_RESULTS --> EVALUATION_COMPLETE
    
    %% Styling
    classDef startEnd fill:#c8e6c9
    classDef process fill:#bbdefb
    classDef decision fill:#ffe0b2
    classDef visualization fill:#e1bee7
    
    class START_EVAL,EVALUATION_COMPLETE startEnd
    class LOAD_MODEL,SET_EVAL_MODE,PROCESS_BATCH,CALC_BASIC,CALC_WEIGHTED,CLASSIFICATION_REPORT,FIND_MISCLASSIFIED,SAVE_RESULTS process
    class MORE_BATCHES decision
    class PLOT_CONFUSION,PLOT_DISTRIBUTION visualization
```

## 🔬 Module Testing Workflow

```mermaid
flowchart TD
    START_TEST([Start Module Testing])
    
    %% Test Setup
    INIT_TESTER[Initialize ModuleTester<br/>List of modules to test]
    
    %% Module Loop
    SELECT_MODULE[Select next module<br/>config, utils, data,<br/>models, train, test]
    
    IMPORT_MODULE[Import module]
    IMPORT_SUCCESS{Import<br/>successful?}
    
    FIND_TEST_FUNC[Find test function<br/>test_[module]_module()]
    FUNC_EXISTS{Test function<br/>exists?}
    
    RUN_TEST[Run test function]
    TEST_SUCCESS{Test<br/>passed?}
    
    RECORD_RESULT[Record test result<br/>PASS/FAIL + details]
    
    MORE_MODULES{More modules<br/>to test?}
    
    %% Summary
    CALC_SUMMARY[Calculate summary<br/>- Total tested<br/>- Passed/Failed counts]
    
    PRINT_DETAILED[Print detailed results<br/>for each module]
    
    ALL_PASSED{All modules<br/>passed?}
    
    PRINT_SUCCESS[🎉 ALL MODULES PASSED! 🎉<br/>Project ready to use]
    PRINT_FAILURE[❌ SOME MODULES FAILED<br/>Check error messages]
    
    TESTING_COMPLETE([Testing Complete])
    
    %% Flow
    START_TEST --> INIT_TESTER
    INIT_TESTER --> SELECT_MODULE
    SELECT_MODULE --> IMPORT_MODULE
    IMPORT_MODULE --> IMPORT_SUCCESS
    
    IMPORT_SUCCESS -->|Yes| FIND_TEST_FUNC
    IMPORT_SUCCESS -->|No| RECORD_RESULT
    
    FIND_TEST_FUNC --> FUNC_EXISTS
    FUNC_EXISTS -->|Yes| RUN_TEST
    FUNC_EXISTS -->|No| RECORD_RESULT
    
    RUN_TEST --> TEST_SUCCESS
    TEST_SUCCESS --> RECORD_RESULT
    
    RECORD_RESULT --> MORE_MODULES
    MORE_MODULES -->|Yes| SELECT_MODULE
    MORE_MODULES -->|No| CALC_SUMMARY
    
    CALC_SUMMARY --> PRINT_DETAILED
    PRINT_DETAILED --> ALL_PASSED
    
    ALL_PASSED -->|Yes| PRINT_SUCCESS
    ALL_PASSED -->|No| PRINT_FAILURE
    
    PRINT_SUCCESS --> TESTING_COMPLETE
    PRINT_FAILURE --> TESTING_COMPLETE
    
    %% Styling
    classDef startEnd fill:#c8e6c9
    classDef process fill:#bbdefb
    classDef decision fill:#ffe0b2
    classDef success fill:#c8e6c9
    classDef failure fill:#ffcdd2
    
    class START_TEST,TESTING_COMPLETE startEnd
    class INIT_TESTER,SELECT_MODULE,IMPORT_MODULE,FIND_TEST_FUNC,RUN_TEST,RECORD_RESULT,CALC_SUMMARY,PRINT_DETAILED process
    class IMPORT_SUCCESS,FUNC_EXISTS,TEST_SUCCESS,MORE_MODULES,ALL_PASSED decision
    class PRINT_SUCCESS success
    class PRINT_FAILURE failure
```

## 🏛️ Model Architecture Overview

```mermaid
graph TB
    %% Input
    INPUT[Input Image<br/>Retinal Fundus Photo]
    
    %% Preprocessing
    PREPROCESS{Preprocessing<br/>Pipeline}
    PIPELINE_A_ARCH[Pipeline A<br/>224×224<br/>- Resize with aspect ratio<br/>- Center on black canvas<br/>- Normalize to ImageNet stats<br/>- Data augmentation]
    PIPELINE_B_ARCH[Pipeline B<br/>299×299<br/>- Resize with aspect ratio<br/>- Center on black canvas<br/>- Normalize to ImageNet stats<br/>- Data augmentation]
    
    %% Models
    VGG16_ARCH[VGG16 Architecture<br/>- Pre-trained backbone<br/>- Frozen early layers<br/>- Custom classifier<br/>- 25M+ parameters]
    
    RESNET_ARCH[ResNet50 Architecture<br/>- Pre-trained backbone<br/>- Skip connections<br/>- Custom final layers<br/>- 23M+ parameters]
    
    INCEPTION_ARCH[InceptionV3 Architecture<br/>- Pre-trained backbone<br/>- Inception modules<br/>- Auxiliary outputs<br/>- 22M+ parameters]
    
    %% Output
    OUTPUT[Output<br/>5-class classification<br/>DR severity levels<br/>0: No DR<br/>1: Mild<br/>2: Moderate<br/>3: Severe<br/>4: Proliferative]
    
    %% Flow
    INPUT --> PREPROCESS
    PREPROCESS -->|VGG16/ResNet50| PIPELINE_A_ARCH
    PREPROCESS -->|InceptionV3| PIPELINE_B_ARCH
    
    PIPELINE_A_ARCH --> VGG16_ARCH
    PIPELINE_A_ARCH --> RESNET_ARCH
    PIPELINE_B_ARCH --> INCEPTION_ARCH
    
    VGG16_ARCH --> OUTPUT
    RESNET_ARCH --> OUTPUT
    INCEPTION_ARCH --> OUTPUT
    
    %% Styling
    classDef input fill:#e3f2fd
    classDef preprocessing fill:#fff3e0
    classDef model fill:#e8f5e8
    classDef output fill:#fce4ec
    
    class INPUT input
    class PREPROCESS,PIPELINE_A_ARCH,PIPELINE_B_ARCH preprocessing
    class VGG16_ARCH,RESNET_ARCH,INCEPTION_ARCH model
    class OUTPUT output
```

## 📊 Data Flow Through System

```mermaid
sequenceDiagram
    participant U as User
    participant M as main.py
    participant C as Config
    participant D as CustomDataLoader
    participant Mo as Models
    participant T as Trainer
    participant E as Evaluator
    participant Ut as Utils
    
    U->>M: Run training pipeline
    M->>C: Load configuration
    C-->>M: Return config object
    
    M->>D: Initialize data loader
    D->>C: Get dataset paths
    D->>D: Load APTOS & EyePACS
    D->>D: Merge and shuffle datasets
    D-->>M: Data loader ready
    
    M->>Mo: Create model (VGG16/ResNet50/InceptionV3)
    Mo-->>M: Return model instance
    
    M->>T: Initialize trainer
    T->>T: Setup optimizer, loss, scheduler
    
    M->>D: Create train/val dataloaders
    D->>D: Apply appropriate pipeline
    D-->>M: Return PyTorch dataloaders
    
    M->>T: Start training
    loop For each epoch
        T->>T: Training phase
        T->>T: Validation phase
        T->>T: Update learning rate
        T->>T: Save best model
    end
    T-->>M: Training complete
    
    M->>E: Initialize evaluator
    M->>D: Create test dataloader
    D-->>M: Return test dataloader
    
    M->>E: Evaluate model
    E->>E: Calculate metrics
    E->>E: Generate plots
    E->>E: Analyze misclassifications
    E-->>M: Return evaluation results
    
    M->>Ut: Save experiment results
    Ut-->>M: Results saved
    
    M-->>U: Training and evaluation complete
```

---

## 📝 How to Use These Flowcharts

1. **Copy the Mermaid code** from any section above
2. **Paste it into a Mermaid editor** like:
   - [Mermaid Live Editor](https://mermaid.live/)
   - GitHub (supports Mermaid in markdown)
   - VS Code with Mermaid extension
   - Any markdown editor with Mermaid support

3. **View the interactive flowchart** to understand the project workflow

## 🎯 Flowchart Purposes

- **Overall Architecture**: High-level view of all modules and their relationships
- **Data Processing**: Detailed workflow of how data flows through the system
- **Training Workflow**: Step-by-step training process
- **Testing & Evaluation**: How model evaluation works
- **Module Testing**: How the testing system validates each component
- **Model Architecture**: Visual representation of the CNN models
- **Data Flow Sequence**: Time-based interaction between components
