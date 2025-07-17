```mermaid
graph TD
    A[Bug Report Input] --> B[Initial Analysis]
    B --> C[AST Parsing & Function Extraction]
    C --> D[Multi-Graph Construction]
    D --> D1[Call Graph Builder]
    D --> D2[Data Flow Graph Builder] 
    D --> D3[Semantic Graph Builder]
    D1 --> E[Graph Merger]
    D2 --> E
    D3 --> E
    E --> F[Primary Function Identification]
    F --> G[Root Cause Analysis]
    G --> G1[Graph Traversal Analysis]
    G --> G2[LLM Reasoning Analysis]
    G --> G3[Pattern Matching Analysis]
    G1 --> H[Candidate Ranking & Fusion]
    G2 --> H
    G3 --> H
    H --> I[Dependency Chain Discovery]
    I --> J[Multi-Agent Planning]
    J --> J1[Planner Agent<br/>Creates fixing sequence]
    J --> J2[Fixer Agent<br/>Generates code changes]
    J --> J3[Validator Agent<br/>Multi-layer validation]
    J --> J4[Coordinator Agent<br/>Orchestrates process]
    J1 --> K[Execution Pipeline]
    J2 --> K
    J3 --> K
    J4 --> K
    K --> L{For Each Function in Dependency Chain}
    L --> M[Generate Fix]
    M --> N[Validation Battery]
    N --> N1[Syntax Validation]
    N --> N2[Type Validation]
    N --> N3[Behavioral Validation]
    N --> N4[Integration Testing]
    N --> N5[Regression Testing]
    N1 --> O{All Validations Pass?}
    N2 --> O
    N3 --> O
    N4 --> O
    N5 --> O
    O -->|Yes| P[Apply Fix]
    O -->|No| Q[Collect Feedback]
    Q --> R[Retry with Feedback]
    R --> M
    P --> S{More Functions in Chain?}
    S -->|Yes| L
    S -->|No| T[Final Integration Test]
    T --> U{Integration Success?}
    U -->|No| V[Backtrack & Replan]
    V --> I
    U -->|Yes| W[Success - Deploy Fix]
    W --> X[Learning System Update]
    X --> Y[Pattern Storage]

    %% High-Contrast Styling with Text Color
    classDef inputOutput fill:#ffcc80,stroke:#bf360c,stroke-width:4px,color:#000000;
    classDef processing fill:#90caf9,stroke:#0d47a1,stroke-width:4px,color:#000000;
    classDef validation fill:#a5d6a7,stroke:#1b5e20,stroke-width:4px,color:#000000;
    classDef decision fill:#ffe082,stroke:#e65100,stroke-width:4px,color:#000000;
    classDef agent fill:#f48fb1,stroke:#880e4f,stroke-width:4px,color:#000000;
    classDef learning fill:#c5e1a5,stroke:#33691e,stroke-width:4px,color:#000000;

    class A,W,Y inputOutput
    class B,C,D,E,F,G,H,I,K,M,P,T,X processing
    class N,N1,N2,N3,N4,N5 validation
    class L,O,S,U decision
    class J1,J2,J3,J4 agent
    class Q,R,V learning
