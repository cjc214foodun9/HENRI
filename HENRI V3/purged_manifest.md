# **Project HENRI Purged Workspace Manifest**

This document outlines the clean-room restructuring of the HENRI-main repository. All diagnostic scaffolding, legacy "V2" deprecated directories, duplicate phase blueprints, and external research PDFs have been purged to eliminate semantic fragmentation and prevent file collision.

## **1\. Purge Register (Deleted Components)**

The following high-entropy, redundant, or obsolete artifacts have been completely purged from the repository:

* **Deprecated Folders:** HENRI V2/DEPRECATED/ and HENRI\_CORE\_V1/ (containing obsolete, competing implementations of the active inference loops).  
* **Duplicate Blueprints:** HENRI Complete Architectural Blueprint.md, HENRI Phase II Architectural Blueprint.md, and any speculative phase roadmaps.  
* **External Non-Code Assets:** External PDFs (What LLM Forecasters Know...pdf), raw audio transcribe logs, and duplicate vast deployment shell scripts.  
* **Scratch Pads:** scratch/ (containing disconnected Kuramoto plotters and master DB scripts that caused configuration drift).  
* **Unscientific Liturgical Plans:** Phase V Weaponization Plan.md (violating first-principles scientific inquiry).

## **2\. Retained & Consolidated Production Layout**

The repository is consolidated into the following minimal, high-efficiency structure:

HENRI-main/  
├── .gitignore                          \# Volatile cache and hardware credentials filter  
├── purged\_manifest.md                  \# This repository mapping and structural audit  
├── henri\_production\_core.py            \# Unified runtime engine (FHRRs, Stiefel, Langevin, Sagnac)  
└── zone\_c\_init.py                      \# Database initialization & pgvector schema generator

This layout enforces strict separation of concerns: henri\_production\_core.py handles continuous-time wave propagation and Langevin relaxation, while zone\_c\_init.py anchors the permanent, zero-entropy axiomatic boundary conditions in Zone C (TimescaleDB).