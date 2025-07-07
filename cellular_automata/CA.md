# Forest Fire Spread Prediction & Cellular Automata Integration

## Comprehensive Requirements & Implementation Guide

### Project Overview

This document summarizes the complete requirements, current state, and implementation strategy for integrating Cellular Automata (CA) fire spread simulation with the existing ML-based fire prediction system for the Uttarakhand forest fire project.

**Timeline**: 2 days for current submission, with future finals preparation
**Evaluation**: ISRO researchers (10-15 years experience) evaluating accuracy and fidelity
**Priority**: Visual appeal + Functional completeness > Technical accuracy (for current submission)

---

## Current System Status

### ML Model & Data Pipeline ✅

- **Model**: ResUNet-A architecture trained and ready
- **Datasets**: Complete stack for April 1 - May 29, 2016
  - DEM (SRTM), ERA5 Daily, LULC 2020, GHSL 2015, Fire masks (VIIRS)
  - Spatial resolution: 30m (aligned)
  - Temporal resolution: Daily (aligned)
  - Geographic coverage: Full Uttarakhand state

### Current Outputs

- **Fire Probability Maps**: 0-1 range, .tif format, 30m resolution
- **Binary Fire/No-fire Masks**: 0 or 1, .tif format, 30m resolution
- **Fire Confidence Zone Maps**: .tif format, 30m resolution
- **Model Location**: `forest_fire_ml/` (functional codebase)
- **Prediction Pipeline**: Sliding window approach in `predict.py`

### Infrastructure

- **Training**: Kaggle GPU notebooks
- **Demo Deployment**: Local hosting (current), cloud deployment (finals)
- **Target Hardware**: Local machines for current submission
- **Performance Expectation**: Full resolution with maximum accuracy (time flexible)

---

## Cellular Automata Requirements

### Core Specifications

- **Spatial Resolution**: 30m (matching ML model output)
- **Geographic Scope**: Full Uttarakhand state with zoom functionality
- **Temporal Resolution**: Hourly simulation updates
- **Simulation Duration**: 1/2/3/6/12 hour scenarios with animation
- **Weather Handling**: Constant daily weather (no hourly interpolation)

### Technical Implementation Strategy

- **Framework**: TensorFlow/GPU acceleration (chosen for finals scalability)
- **Input Source**: Daily ML-generated probability maps (not real-time inference)
- **Physics Model**: Simplified rules (current), Rothermel physics (finals)
- **Natural Barriers**: GHSL maps for roads, rivers, built areas
- **Fire Suppression**: Basic implementation using barrier data

### CA Algorithm Design

```
Input: Daily probability map (0-1), weather constants, ignition points
Process:
1. Initialize grid from ML probability map
2. Apply CA rules with neighborhood analysis
3. Factor in wind direction/speed (constant daily)
4. Apply barrier effects (GHSL-derived)
5. Update grid state hourly
6. Generate animation frames
Output: Hourly fire spread maps + animation
```

---

## Website Integration Plan

### Development Strategy: Option B (Parallel Development)

**Day 1**:

- Person 1: CA core implementation + basic testing
- Person 2: Website skeleton + basic interface
  **Day 2**:
- Integration sprint + optimization + debugging

### User Interface Specifications

#### Current Submission Features (Basic)

- **Ignition Interface**: Click points on map to ignite fires
- **Date Selection**: Choose from available dates (April-May 2016)
- **Animation Viewer**: Watch pre-computed simulation with time controls
- **Map Interaction**: Zoom, pan, basic map controls
- **Output Display**: Side-by-side comparison (probability vs. simulated spread)

#### Future Features (Finals)

- Draw custom fire perimeters
- Real-time weather parameter adjustment
- Multi-scenario comparison
- Export simulation results
- Advanced visualization controls

### Technology Stack

- **Current**: Local web application
- **Future**: Online deployment-ready architecture
- **Integration**: Direct connection between CA engine and web interface
- **Data Flow**: ML predictions → CA simulation → Web visualization

---

## Data Flow Architecture

```
Workflow Pipeline:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Daily Stacked   │ -> │ ML Model        │ -> │ Probability     │
│ Input Data      │    │ (ResUNet-A)     │    │ Maps (.tif)     │
│ (9 bands)       │    │                 │    │ (0-1 range)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Website         │ <- │ CA Simulation   │ <- │ User Ignition   │
│ Animation       │    │ Engine          │    │ + Date Select   │
│ Display         │    │ (TensorFlow)    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### File Organization

```
cellular_automata/
├── CA.md (this file)
├── ca_engine/
│   ├── core.py (main CA implementation)
│   ├── rules.py (fire spread rules)
│   ├── utils.py (helper functions)
│   └── config.py (simulation parameters)
├── web_interface/
│   ├── app.py (main web application)
│   ├── static/ (CSS, JS, images)
│   ├── templates/ (HTML templates)
│   └── api/ (backend API endpoints)
└── integration/
    ├── data_bridge.py (ML model ↔ CA interface)
    ├── simulation_runner.py (orchestration)
    └── output_handler.py (results management)
```

---

## Submission Requirements

### Deliverables

1. **Fire Prediction Maps**: Next-day probability maps
2. **Animated Fire Spread**: 1/2/3/6/12 hour simulations
3. **Interactive Website**: Local demo with basic functionality
4. **Clean Code + Documentation**: Primary evaluation criteria
5. **Optional**: Live interactive demo (good to have)

### Evaluation Metrics

- **Accuracy**: Fire prediction map quality
- **Fidelity**: Spread simulation realism
- **Visual Appeal**: Demo presentation quality
- **Functional Completeness**: Feature implementation

---

## Technical Decisions & Constraints

### Resolved Questions

- **Model Output**: Use daily probability maps as CA input (not real-time inference)
- **Weather Data**: Constant daily values (no hourly interpolation needed)
- **Spatial Coverage**: Full Uttarakhand state with zoom capability
- **Resolution**: Maintain 30m throughout pipeline
- **Performance**: Accuracy over speed for current submission
- **Physics**: Simplified rules now, advanced physics for finals
- **Framework**: TensorFlow/GPU for scalability

### Implementation Priorities

1. **Visual Appeal**: Impressive demo for submission
2. **Functional Completeness**: All basic features working
3. **Technical Accuracy**: Realistic but not necessarily scientifically perfect

---

## Next Steps

### Immediate Actions

1. **Verify Current Outputs**: Test prediction pipeline on 2016 data
2. **CA Core Development**: Implement basic TensorFlow-based CA engine
3. **Web Framework Setup**: Create basic interface skeleton
4. **Integration Planning**: Design data flow between components

### Development Milestones

- **Day 1 Morning**: CA core + web skeleton
- **Day 1 Evening**: Basic functionality testing
- **Day 2 Morning**: Integration + debugging
- **Day 2 Evening**: Demo preparation + documentation

---

## Risk Assessment & Mitigation

### High Risk Areas

- **TensorFlow CA Implementation**: Complex but necessary for finals
- **Real-time Integration**: ML model → CA → Web pipeline
- **Performance Optimization**: Full-state simulation at 30m resolution

### Fallback Strategies

- **Pre-computed Scenarios**: If real-time simulation fails
- **Simplified CA Rules**: If TensorFlow implementation is too complex
- **Static Visualization**: If interactive features don't integrate properly

---

## Future Enhancements (Finals)

### Advanced Features

- **Rothermel Physics**: Scientific fire spread equations
- **Real-time Weather**: API integration for current conditions
- **Fire Suppression Modeling**: Aircraft, ground crews, water sources
- **Multi-scenario Analysis**: Parallel simulation comparison
- **Export Capabilities**: Results download and sharing

### Scalability Considerations

- **Cloud Deployment**: ISRO-provided infrastructure
- **Multi-user Support**: Concurrent simulation handling
- **Data Pipeline Optimization**: Faster prediction generation
- **Advanced Visualization**: 3D terrain, satellite overlay

---

**Document Status**: Complete requirements analysis
**Last Updated**: July 6, 2025
**Next Review**: After initial implementation testing
