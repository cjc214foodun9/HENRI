import torch

def execution_loop(user_prompt, orchestrator, rehypothecator, zone_b_emulator, max_bounces=10):
    """
    Orchestrates the closed-loop, multi-agent cognitive cycle between the CPU 7B Swarm,
    the L3 Router, the Zone B BTO crystal emulator, and the telemetry feedback.
    """
    print(f"\n--- Booting Closed-Loop Cognitive Cycle: '{user_prompt}' ---")
    current_prompt = user_prompt
    latent_shift = None
    
    for cycle in range(max_bounces):
        print(f"\n[Cycle {cycle + 1}/{max_bounces}] Starting cognitive iteration...")
        
        # 1. Ingest prompt and evaluate master routing using L3 Router (first token/prompt representation)
        # We project the text prompt to get the optimal routing directive
        # (For the mock, orchestrator uses the router to decide which 7B sub-agent to wake up)
        with torch.no_grad():
            # Let's tokenize user prompt (using mock embedding/tokens in mock orchestrator)
            mock_tokens = orchestrator.tokenize(current_prompt)
            _, winning_master_id, resonance_scores = orchestrator.router_model(tokens=mock_tokens)
            master_id = winning_master_id[0].item()
            print(f"  -> L3 Router routing directive: Master Signature {master_id} (Resonance: {resonance_scores[0, master_id].item():.4f})")
            
        # 2. Dispatch payload to the selected 7B Swarm Master in system RAM
        # We pass the latent_shift (the error vector feedback) to steer model generation
        response = orchestrator.dispatch_payload(
            master_id=master_id, 
            prompt=current_prompt, 
            latent_shift=latent_shift
        )
        proposed_text = response['text']
        print(f"  -> 7B Master {master_id} proposed output (first 80 chars): '{proposed_text[:80]}...'")
        
        # 3. Translate the proposed discrete text into the 4096-D continuous wavefront
        with torch.no_grad():
            proposed_tokens = orchestrator.tokenize(proposed_text)
            proposed_hrr_wave, _, _ = orchestrator.router_model(tokens=proposed_tokens)
            
        # 4. Fire the wavefront into the Zone B crystal emulator (Physics Engine)
        # Sagnac logic veto evaluates constructive/destructive interference against Zone C attractors
        sagnac_delta, epiplexity_score, error_vector = zone_b_emulator.fire(proposed_hrr_wave)
        print(f"  -> Zone B Crystal Telemetry: Sagnac Delta: {sagnac_delta:.4f} | Epiplexity Score: {epiplexity_score:.4f}")
        
        # 5. Evaluate the physical telemetry via the Rehypothecator
        telemetry_directive = rehypothecator.evaluate_telemetry(
            current_hrr_wave=proposed_hrr_wave,
            sagnac_delta=sagnac_delta,
            epiplexity_score=epiplexity_score
        )
        
        # 6. Apply governor action
        action = telemetry_directive["action"]
        
        if action == "TRANSLATE_TO_USER":
            # Coherence achieved. Attractor collapsed.
            print(f"[+] Resonance Achieved at Cycle {cycle + 1}. Returning verified result.")
            # Log successful convergence to system ledger
            orchestrator.log_thermodynamic_ledger(
                prompt=user_prompt,
                cycles=cycle + 1,
                sagnac_delta=sagnac_delta,
                epiplexity=epiplexity_score,
                status="CONVERGED"
            )
            return proposed_text
            
        elif action in ["RETRY_WITH_HEAT", "HARD_RESET_AND_RETRY"]:
            # System is in a Logic Lock or has triggered a Veto. Adjust physical microheaters.
            heat = telemetry_directive["langevin_heat"]
            zone_b_emulator.set_microheaters(heat)
            
            # Feed back the error gradient (latent_shift) to reshape next speculative generation
            latent_shift = telemetry_directive["vector_shift"]
            
            # Append prompt injection to discrete instruction context
            current_prompt = f"{user_prompt}\n\n{telemetry_directive['prompt_injection']}"
            print(f"  -> Action: {action}. Applied Langevin Heat: {heat:.2f}V. Steered latent vectors.")
            
            # Log iteration to system ledger
            orchestrator.log_thermodynamic_ledger(
                prompt=user_prompt,
                cycles=cycle + 1,
                sagnac_delta=sagnac_delta,
                epiplexity=epiplexity_score,
                status=action
            )
            
    # If loop exhausts bounces without achieving resonance
    print("[!] Cognitive cycle exhausted without achieving resonance.")
    orchestrator.log_thermodynamic_ledger(
        prompt=user_prompt,
        cycles=max_bounces,
        sagnac_delta=sagnac_delta,
        epiplexity=epiplexity_score,
        status="TIMEOUT"
    )
    return "ERROR: System exhausted thermal budget without finding geometric resonance."
