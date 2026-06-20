import os
import psycopg

db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:password@localhost:5433/henri")

try:
    print("[*] Connecting to TimescaleDB...")
    with psycopg.connect(db_url, connect_timeout=3) as conn:
        with conn.cursor() as cur:
            conn.autocommit = True
            
            # 1. Create stirrup_motor_command_registry
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stirrup_motor_command_registry (
                    motor_id INT PRIMARY KEY,
                    command_string TEXT NOT NULL,
                    command_type TEXT NOT NULL,
                    description TEXT
                );
            """)
            print("[+] Table 'stirrup_motor_command_registry' verified/created.")
            
            # Seed default motor commands
            default_commands = [
                (0, "API_CALL: actuator.stabilize_gripper_torque()", "API_CALL", "Stabilizes gripper torque to target boundary"),
                (1, "EXEC_SHELL: cd /workspace/hardware && ./step_motor_vulkan --axis_x 12", "EXEC_SHELL", "Steps vulkan motor on X axis"),
                (2, "API_CALL: scada.alleviate_fluid_valve_pressure()", "API_CALL", "Alleviates valve pressure via fluid mechanics"),
                (3, "EXEC_SHELL: python firmware_reflash_probe.py --mode homeostatic", "EXEC_SHELL", "Reflashes probe firmware in homeostatic mode"),
                (4, "API_CALL: robot_arm.engage_four_wave_mixer_conjugation()", "API_CALL", "Engages four wave mixer conjugation")
            ]
            
            for motor_id, cmd, cmd_type, desc in default_commands:
                cur.execute("""
                    INSERT INTO stirrup_motor_command_registry (motor_id, command_string, command_type, description)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (motor_id) DO NOTHING;
                """, (motor_id, cmd, cmd_type, desc))
            print("[+] Seeded default commands in stirrup_motor_command_registry.")
            
            # 2. Create stirrup_telemetry_ledger
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stirrup_telemetry_ledger (
                    timestamp TIMESTAMPTZ NOT NULL,
                    inference_id UUID NOT NULL,
                    selected_plan_index INT NOT NULL,
                    thermodynamic_stress_cost FLOAT NOT NULL,
                    sigreg_disentanglement_score FLOAT NOT NULL,
                    transduced_motor_token_id INT NOT NULL,
                    actuated_command TEXT NOT NULL,
                    success BOOLEAN NOT NULL
                );
            """)
            print("[+] Table 'stirrup_telemetry_ledger' verified/created.")
            
            # Convert to hypertable
            try:
                cur.execute("SELECT create_hypertable('stirrup_telemetry_ledger', 'timestamp');")
                print("[+] stirrup_telemetry_ledger successfully converted to hypertable.")
            except Exception as e:
                if "already a hypertable" not in str(e).lower():
                    print(f"[WARNING] hypertable creation failed: {e}")
                    
            print("[SUCCESS] Stirrup TimescaleDB migrations successfully completed.")
            
except Exception as e:
    print("[ERROR] Failed to run database extension:", e)
