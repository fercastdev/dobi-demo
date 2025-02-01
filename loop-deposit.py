import time
from web3 import Web3
import json
# --- Configuration: Update these with your own values ---

with open("config2.json", "r") as f:
    config = json.load(f)

RPC_URL = config["RPC_URL"]
CONTRACT_ADDRESS = config["CONTRACT_ADDRESS"]
PRIVATE_KEY = config["PRIVATE_KEY"]
ACCOUNT_ADDRESS = config["ACCOUNT_ADDRESS"]


AMOUNT_ETH_DEPOSIT = config["AMOUNT_ETH_DEPOSIT"]

# --- Minimal ABI containing only the deposit() function ---
CONTRACT_ABI = [
    {
        "inputs": [],
        "name": "deposit",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function"
    }
]

# --- Connect to the blockchain ---
web3 = Web3(Web3.HTTPProvider(RPC_URL))
if not web3.is_connected():
    print("Failed to connect to the blockchain!")
    exit()

# Ensure addresses are in checksum format.
account = Web3.to_checksum_address(ACCOUNT_ADDRESS)
contract = web3.eth.contract(
    address=Web3.to_checksum_address(CONTRACT_ADDRESS),
    abi=CONTRACT_ABI
)

print("Starting deposit loop...")

while True:
    try:
        # Get the nonce for the account
        nonce = web3.eth.get_transaction_count(account)

        # Build a transaction calling deposit() with a value of 1 ETH.
        txn = contract.functions.deposit().build_transaction({
            "from": account,
            "value": web3.to_wei(AMOUNT_ETH_DEPOSIT, "ether"),
            "gas": 100000000,
            "gasPrice": web3.eth.gas_price,
            "nonce": nonce,
        })

        # Sign the transaction with the private key.
        signed_txn = web3.eth.account.sign_transaction(txn, PRIVATE_KEY)

        # Send the signed transaction.
        tx_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)
        print(f"Deposited {AMOUNT_ETH_DEPOSIT} ETH. Transaction hash:", web3.to_hex(tx_hash))

    except Exception as e:
        print("Error during deposit:", e)

    # Wait for 5 minutes (300 seconds) before the next deposit.
    time.sleep(300)