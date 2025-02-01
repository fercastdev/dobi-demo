import time
from typing import Any, Callable, Dict
import json

# ---- Pydantic v2 imports ----
from pydantic import BaseModel, Field, PrivateAttr, model_validator

# ---- LangChain & LangGraph imports ----
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, END

# ---- Web3 imports ----
from web3 import Web3

import requests

with open("config.json", "r") as f:
    config = json.load(f)

RPC_URL = config["RPC_URL"]
CONTRACT_ADDRESS = config["CONTRACT_ADDRESS"]
PRIVATE_KEY = config["PRIVATE_KEY"]
TARGET_ADDRESS = config["TARGET_ADDRESS"]

WEBHOOK_URL =  config["WEBHOOK_URL"] # 
API_KEY =  config["API_KEY"] # 

CONTRACT_ADDRESS = config["CONTRACT_ADDRESS"] #  ""




DOBI_HEADERS = {
  'Content-Type': 'application/json',
  'x-api-key': API_KEY,
  'API_KEY': API_KEY
}


def log_to_webhook(message: str, data: dict = None):
    """
    Posts a JSON payload to the specified webhook URL.
    """
    payload = {"message": message, "data": data}
    try:
        response = requests.request("POST",WEBHOOK_URL, json=payload,  headers=DOBI_HEADERS)
        if response.status_code == 200:
            print("Logged to webhook successfully.")
        else:
            print(f"Webhook responded with status code: {response.status_code}")
    except Exception as e:
        print("Error sending to webhook:", e)

###############################################
# 1) Define State and Condition Checking
###############################################
class ContractMonitorState(BaseModel):
    """Tracks if condition is met and stores relevant event data."""
    condition_met: bool = False
    event_data: Dict[str, Any] = Field(default_factory=dict)

def condition_check(event) -> bool:
    """Return True if the deposit amount (in wei) is > 0.1 ETH."""
    return event["args"]["amount"] > Web3.to_wei(0.1, "ether")


###############################################
# 2) Contract Monitor Tool
###############################################
class SmartContractMonitorTool(BaseTool):
    name: str = Field(default="smart_contract_monitor")
    description: str = Field(default="Monitors a contract for new deposit events.")

    rpc_url: str
    contract_address: str
    contract_abi: Any
    condition_fn: Callable[[Any], bool]

    _web3: Web3 = PrivateAttr()
    _contract: Any = PrivateAttr()
    _filter: Any = PrivateAttr(default=None)
    _last_checked_block: int = PrivateAttr(default=None)

    @model_validator(mode="after")
    def init_runtime_attrs(self):
        self._web3 = Web3(Web3.HTTPProvider(self.rpc_url))
        checksum = Web3.to_checksum_address(self.contract_address)
        self._contract = self._web3.eth.contract(address=checksum, abi=self.contract_abi)
        self._last_checked_block = self._web3.eth.block_number

        return self


    def _run(self, state: ContractMonitorState) -> ContractMonitorState:
        print("Checking for new `DepositMade` events...")

        
        current_block = self._web3.eth.block_number
        threshold = 60  # Approximate number of blocks in 10 minutes
        
        # Only check for events if enough blocks have passed since the last check.
        if current_block - self._last_checked_block < threshold:
            print(f"Only {current_block - self._last_checked_block} blocks passed; waiting for at least {threshold} blocks.")
            log_to_webhook("SmartContractMonitorTool output", {"contract address": self.contract_address, "status": f"Only {current_block - self._last_checked_block} blocks passed; waiting for at least {threshold} blocks."})

            return state
        log_to_webhook("SmartContractMonitorTool output", {"contract address": self.contract_address, "status": "Checking for new `DepositMade` events..."})

        from_block = self._last_checked_block + 1
        # Update the last checked block to the current block.
        self._last_checked_block = current_block
        
        # Create a filter for events from from_block to the current/latest block.
        event_filter = self._contract.events.DepositMade.create_filter(
            from_block=from_block,
            to_block='latest'
        )
        if self._filter is None:
            self._filter = event_filter

        try:
            new_events = self._filter.get_new_entries()
        except ValueError:
            # In case the filter goes stale, recreate it.
            self._filter = event_filter
            new_events = self._filter.get_new_entries()
            
        print("checking events")

        for event in new_events:
            print(event)
            if self.condition_fn(event):
                # Convert the deposit amount from wei to ETH for clarity.
                deposit_wei = event["args"]["amount"]
                deposit_eth = self._web3.from_wei(deposit_wei, "ether")
                
                # Prepare event data with both raw and converted values.
                event_data = dict(event)
                args_dict = dict(event_data["args"])

                # Now you can safely add a new field.
                args_dict["amount_eth"] = deposit_eth
                event_data["args"] = args_dict

                print(f">>> Condition MET: deposit exceeds 0.1 ETH (Deposit: {deposit_eth} ETH)")
                log_to_webhook("SmartContractMonitorTool output", {"contract address": self.contract_address, "status": f">>> Condition MET: deposit exceeds 0.1 ETH (Deposit: {deposit_eth} ETH)"})
                return ContractMonitorState(condition_met=True, event_data=event_data)
        return ContractMonitorState(condition_met=False, event_data={})




###############################################
# 3) Agent Withdraw Tool
###############################################
class AgentWithdrawTool(BaseTool):
    """
    When condition_met is True, the agent (contract owner) calls the contract’s withdraw() function.
    The target address is supplied by the agent (via the Python script).
    """
    name: str = Field(default="agent_withdraw")
    description: str = Field(default="Calls the contract withdraw function to forward funds to a specified target address.")

    rpc_url: str
    private_key: str
    contract_address: str
    contract_abi: Any
    target_address: str  # Provided externally by the agent

    _web3: Web3 = PrivateAttr()
    _contract: Any = PrivateAttr()
    _agent: str = PrivateAttr()

    @model_validator(mode="after")
    def init_runtime_attrs(self):
        self._web3 = Web3(Web3.HTTPProvider(self.rpc_url))
        self._agent = self._web3.eth.account.from_key(self.private_key).address
        checksum = Web3.to_checksum_address(self.contract_address)
        self._contract = self._web3.eth.contract(address=checksum, abi=self.contract_abi)
        return self

    def _run(self, state: ContractMonitorState) -> ContractMonitorState:
        if not state.condition_met:
            print("AgentWithdrawTool: condition not met, skipping withdrawal.")
            return state

        print(f"AgentWithdrawTool: Agent {self._agent} calling withdraw() on contract {self._contract.address} with target {self.target_address} ...")
        nonce = self._web3.eth.get_transaction_count(self._agent)
        txn = self._contract.functions.withdraw(
            Web3.to_checksum_address(self.target_address)
        ).build_transaction({
            "from": self._agent,
            "nonce": nonce,
            "gas": 2000000000,
            "gasPrice": self._web3.eth.gas_price,
        })
        signed_txn = self._web3.eth.account.sign_transaction(txn, self.private_key)
        tx_hash = self._web3.eth.send_raw_transaction(signed_txn.raw_transaction)
        tx_hash_hex = self._web3.to_hex(tx_hash)

        print(f"Withdraw transaction sent! Tx Hash: {tx_hash_hex}")
        log_to_webhook("AgentWithdrawTool output", {"agent": self._agent, "target": self.target_address, "tx_hash": tx_hash_hex})

        state.condition_met = False  # Reset state after withdrawal
        return state

##################################################################
# 4. LLMDecisionTool
##################################################################

from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from langchain.llms import Ollama  # Provided by langchain-ollama

###############################################
class OllamaDecisionTool(BaseTool):
    """
    Example: Takes the monitor's event data and asks Ollama if we should
    proceed with the transfer. The user must have Ollama running locally
    with the model 'deepseek-r1:8b' installed.
    """
    name: str = Field(default="ollama_decision_tool")
    description: str = Field(default="Uses Ollama LLM to decide whether to proceed with ETH transfer.")

    # No special user fields needed, but you can add them if needed
    _ollama: Ollama = PrivateAttr()

    @model_validator(mode="after")
    def init_ollama(self):
        # Create an Ollama instance pointing to your local server
        # If you have a custom port or URL, specify base_url="http://localhost:11411"
        self._ollama = Ollama(model="deepseek-r1:8b")
        return self

    def _run(self, state: ContractMonitorState) -> ContractMonitorState:
        if not state.condition_met:
            print("OllamaDecisionTool: condition not met, skipping LLM check.")
            return state

        prompt = f"""
    We have a deposit event:
    Deposit amount: {state.event_data.get("args", {}).get("amount_eth", "Unknown")} ETH.
    Other event details:
    {state.event_data}

    Based solely on the above event details, if this deposit is greater than 0.1 ETH, respond with exactly one word in the final line: YES or NO.
    Any chain-of-thought can appear above, but the very last line must be exactly YES or NO.
"""
        response = self._ollama(prompt).strip().upper()
        print(f"Ollama says: {response}")

        # If the LLM says "YES", we keep condition_met=True; otherwise false.
        
        lines = response.strip().split('\n')
        final_line = lines[-1].strip().upper()
        decision = final_line.endswith("YES")
        log_to_webhook("OllamaDecisionTool output", {"raw_response": response, "final_decision": final_line})

        state.condition_met = decision
        return state


DOB_MONITOR_ABI = [
        {
            "anonymous": False,
            "inputs": [
                {"indexed": True, "internalType": "address", "name": "sender", "type": "address"},
                {"indexed": False, "internalType": "uint256", "name": "amount", "type": "uint256"}
            ],
            "name": "DepositMade",
            "type": "event"
        },
        {
            "anonymous": False,
            "inputs": [
                {"indexed": True, "internalType": "address", "name": "target", "type": "address"},
                {"indexed": False, "internalType": "uint256", "name": "amount", "type": "uint256"}
            ],
            "name": "Withdrawal",
            "type": "event"
        },
        {
            "inputs": [],
            "name": "deposit",
            "outputs": [],
            "stateMutability": "payable",
            "type": "function"
        },
        {
            "inputs": [
                {"internalType": "address", "name": "_target", "type": "address"}
            ],
            "name": "withdraw",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "getBalance",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function"
        }
    ]


##################################################################
# 5. Build and Run the LangGraph Agent Workflow
##################################################################
def build_workflow(
    rpc_url: str,
    contract_address: str,
    contract_abi: Any,
    private_key: str,
    target_address: str
):
    monitor_tool = SmartContractMonitorTool(
        rpc_url=rpc_url,
        contract_address=contract_address,
        contract_abi=contract_abi,
        condition_fn=condition_check
    )
    decision_tool = OllamaDecisionTool()
    withdraw_tool = AgentWithdrawTool(
        rpc_url=rpc_url,
        private_key=private_key,
        contract_address=contract_address,
        contract_abi=contract_abi,
        target_address=target_address  
    )

    workflow = StateGraph(ContractMonitorState)
    workflow.add_node("monitor", monitor_tool._run)
    workflow.add_node("decision", decision_tool._run)
    workflow.add_node("withdraw", withdraw_tool._run)

    # Graph: monitor -> decision -> withdraw -> END
    workflow.add_edge("monitor", "decision")
    workflow.add_edge("decision", "withdraw")
    workflow.add_edge("withdraw", END)

    workflow.set_entry_point("monitor")
    return workflow.compile()


if __name__ == "__main__":
    # Connect to local Hardhat


    # Build the app
    app = build_workflow(
        rpc_url=RPC_URL,
        contract_address=CONTRACT_ADDRESS,
        contract_abi=DOB_MONITOR_ABI,
        private_key=PRIVATE_KEY,
        target_address=TARGET_ADDRESS
        #amount_eth=AMOUNT_ETH
    )

    state = ContractMonitorState()

    print("Starting LangGraph loop. Monitoring local Hardhat contract...\n")
    while True:
        # Run one cycle
        state = app.invoke(state)
        # Sleep 5s before checking again
        time.sleep(30)