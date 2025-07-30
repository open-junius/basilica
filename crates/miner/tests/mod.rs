// mod integration;
// mod unit;

#[cfg(test)]
mod tests {
    use alloy::signers::local::PrivateKeySigner;
    use alloy::signers::Signer;
    use alloy_primitives::{address, FixedBytes, U256};
    use alloy_provider::ProviderBuilder;
    use alloy_sol_types::SolEvent;
    use collateral::Collateral::{self, CollateralInstance};
    use miner::config::MinerConfig;
    use rand::Rng;

    /// export BASILCA_MINER_CONTRACT__PRIVATE_KEY="0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
    /// 0xa8b2b82247e3f2b49ee8858b088405e35755c096 deployed in testnet
    /// chain id is 945, minCollateralIncrease is 1, decisionTimeout is 1, trustee is 0xABCaD56aa87f3718C8892B48cB443c017Cd632BB
    ///
    /// 0x119ecacb1322cd9d581d550b52e199ec97a33e2e
    /// decisionTimeout is 20 for the reclaim deny testing
    async fn get_contract() -> anyhow::Result<CollateralInstance<impl alloy_provider::Provider>> {
        let config = MinerConfig::load()?;
        let rpc_url = "https://test.finney.opentensor.ai";
        let private_key = config.miner_contract.private_key;
        let contract_address = address!("0xa8b2b82247e3f2b49ee8858b088405e35755c096");

        let mut signer: PrivateKeySigner = private_key.parse().unwrap();
        signer.set_chain_id(Some(945));

        let provider = ProviderBuilder::new()
            .wallet(signer)
            .connect(rpc_url)
            .await?;

        let contract = Collateral::new(contract_address, provider);
        Ok(contract)
    }

    #[tokio::test]
    #[ignore]
    // cargo test --package miner --test mod -- tests::test_deposit --exact --nocapture
    async fn test_deposit() -> anyhow::Result<()> {
        let contract = get_contract().await?;
        println!("trustee: {:?}", contract.TRUSTEE().call().await.unwrap());
        println!("netuid: {:?}", contract.NETUID().call().await.unwrap());
        println!(
            "decision_timeout: {:?}",
            contract.DECISION_TIMEOUT().call().await.unwrap()
        );
        println!(
            "min_collateral_increase: {:?}",
            contract.MIN_COLLATERAL_INCREASE().call().await.unwrap()
        );

        let executor_id: u128 = rand::thread_rng().gen_range(0..10000000000);
        let amount = U256::from(10);
        let deposit_tx = contract.deposit(executor_id.into()).value(amount);
        let deposit_tx_receipt = deposit_tx.send().await?.get_receipt().await?;

        let mut deposit_found = false;
        deposit_tx_receipt.logs().iter().for_each(|log| {
            if let Ok(event) = Collateral::Deposit::decode_log(&log.inner) {
                assert!(FixedBytes::from(executor_id) == event.executorId);
                deposit_found = true;
            }
        });
        assert!(deposit_found);

        let collaterals = contract
            .collaterals(executor_id.into())
            .call()
            .await
            .unwrap();

        assert_eq!(collaterals, amount);

        Ok(())
    }

    #[tokio::test]
    #[ignore]
    // cargo test --package miner --test mod -- tests::test_reclaim_finalize --exact --nocapture
    async fn test_reclaim_finalize() -> anyhow::Result<()> {
        let contract = get_contract().await?;

        // Deposit first
        let executor_id: u128 = rand::thread_rng().gen_range(0..10000000000);
        let amount = U256::from(10);
        let deposit_tx = contract.deposit(executor_id.into()).value(amount);
        let _deposit_tx_receipt = deposit_tx.send().await?.get_receipt().await?;

        // Start reclaim process
        let url = "example.com";
        let url_checksum = 123_u128;
        let reclaim_tx =
            contract.reclaimCollateral(executor_id.into(), url.to_owned(), url_checksum.into());
        let reclaim_receipt = reclaim_tx.send().await?.get_receipt().await?;

        let mut reclaim_id = U256::from(0);
        reclaim_receipt.logs().iter().for_each(|log| {
            if let Ok(event) = Collateral::ReclaimProcessStarted::decode_log(&log.inner) {
                reclaim_id = event.reclaimRequestId;
            }
        });

        tokio::time::sleep(std::time::Duration::from_secs(30)).await;

        // Test denyReclaimRequest (trustee only)
        let finalize_tx = contract.finalizeReclaim(reclaim_id);
        let finalize_receipt = finalize_tx.send().await?.get_receipt().await?;
        let mut finalize_found = false;
        // Check for Denied event
        finalize_receipt.logs().iter().for_each(|log| {
            if let Ok(event) = Collateral::Reclaimed::decode_log(&log.inner) {
                assert_eq!(event.reclaimRequestId, reclaim_id);
                finalize_found = true;
            }
        });
        assert!(finalize_found);
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    // cargo test --package miner --test mod -- tests::test_reclaim_deny --exact --nocapture
    // to test deny, need set the decision timeout to a bigger number
    async fn test_reclaim_deny() -> anyhow::Result<()> {
        let contract = get_contract().await?;

        let executor_id: u128 = rand::thread_rng().gen_range(0..10000000000);
        let amount = U256::from(10);
        let deposit_tx = contract.deposit(executor_id.into()).value(amount);
        let _deposit_tx_receipt = deposit_tx.send().await?.get_receipt().await?;

        // Start reclaim process
        let url = "example.com";
        let url_checksum = 123_u128;
        let reclaim_tx =
            contract.reclaimCollateral(executor_id.into(), url.to_owned(), url_checksum.into());
        let reclaim_receipt = reclaim_tx.send().await?.get_receipt().await?;

        let mut reclaim_id = U256::from(0);
        reclaim_receipt.logs().iter().for_each(|log| {
            if let Ok(event) = Collateral::ReclaimProcessStarted::decode_log(&log.inner) {
                reclaim_id = event.reclaimRequestId;
            }
        });

        // Test denyReclaimRequest (trustee only)
        let deny_tx = contract.denyReclaimRequest(reclaim_id, url.to_owned(), url_checksum.into());
        let deny_receipt = deny_tx.send().await?.get_receipt().await?;

        // Check for Denied event
        let mut denied_found = false;
        deny_receipt.logs().iter().for_each(|log| {
            if let Ok(event) = Collateral::Denied::decode_log(&log.inner) {
                assert_eq!(event.reclaimRequestId, reclaim_id);
                denied_found = true;
            }
        });
        assert!(denied_found);
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    // cargo test --package miner --test mod -- tests::test_slash --exact --nocapture
    async fn test_slash() -> anyhow::Result<()> {
        let contract = get_contract().await?;

        let executor_id: u128 = rand::thread_rng().gen_range(0..10000000000);
        let amount = U256::from(10);
        let deposit_tx = contract.deposit(executor_id.into()).value(amount);
        let _deposit_tx_receipt = deposit_tx.send().await?.get_receipt().await?;

        // Start reclaim process
        let url = "example.com";
        let url_checksum = 123_u128;
        let slash_tx =
            contract.slashCollateral(executor_id.into(), url.to_owned(), url_checksum.into());
        let slash_receipt = slash_tx.send().await?.get_receipt().await?;

        slash_receipt.logs().iter().for_each(|log| {
            if let Ok(event) = Collateral::Slashed::decode_log(&log.inner) {
                assert_eq!(event.executorId, FixedBytes::from(executor_id));
            }
        });
        Ok(())
    }
}
