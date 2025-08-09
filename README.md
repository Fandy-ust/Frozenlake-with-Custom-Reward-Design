# From Failure to Perfection: Solving 8x8 FrozenLake with Custom Reward Design

## The Journey

### Phase 1: Basic Understanding (4x4 Success)

In the initial exploration phase of reinforcement learning, we first opted for the classic 4x4 grid version of the `FrozenLake-v1` environment. This environment is renowned for its relatively small state space (16 discrete states), making it an ideal platform for understanding fundamental algorithms like Q-learning.

During this phase, we employed a single Q-learning agent and adhered to the environment's **default sparse reward setting**: a positive reward was granted only when the agent successfully reached the goal ('G'), while all other state transitions (including falling into a hole 'H' or staying on a frozen surface 'F') yielded zero reward. Training involved moderate discount factor (gamma) and learning rate (alpha), complemented by a standard epsilon-greedy exploration policy.

Thanks to the compact nature of the 4x4 map, even with the agent's initial random exploration, it was relatively easy to accidentally reach the goal over a large number of episodes and extended exploration steps. Each such success triggered the Q-learning's **Temporal Difference (TD) update mechanism**. As the number of successful goal arrivals increased, the reward signal propagated backward like ripples, effectively updating the Q-values of all relevant state-action pairs along the path from the start to the goal. This enabled the agent to learn a stable policy for reaching the target. The success of this phase validated our understanding of reinforcement learning fundamentals and laid the groundwork for tackling more complex challenges.

### Phase 2: The 8x8 Challenge (Complete Failure)

Encouraged by our success in the 4x4 environment, we confidently migrated the algorithm to the **8x8 `FrozenLake-v1` environment**. However, the state space dimension dramatically expanded from 16 to **64 states**, and the number of "holes" in the environment significantly increased, escalating the problem's complexity exponentially.

Directly applying the effective Q-learning strategy from the 4x4 environment to the 8x8 version resulted in a frustrating "complete failure." Due to the exponential growth of the state space and the presence of numerous dangerous holes, the agent was almost never able to reach the goal by pure random exploration. In the vast majority of cases, the agent would fall into a hole along its exploratory path, sometimes even near the starting point, ending the current episode prematurely. This led to the agent being **"stuck" midway**, completely unable to receive any reward signals. This phenomenon starkly revealed the severity of the **sparse reward** problem in the 8x8 environment.

### Phase 3: Problem Diagnosis - Sparse Reward Trap

Following the failure in Phase 2, we conducted an in-depth problem diagnosis. The core issue was evident: the **reward signal could barely propagate to the agent's current state**. Whether the agent never reached the goal, or only occasionally reached it by pure chance (which was extremely rare in our experiments), the reward obtained at the goal was simply not effectively or timely back-propagated to the Q-values of most states along the path.

During actual code execution, we observed that the agent almost never successfully reached the goal. After **10,000 training episodes, the success rate remained a striking 0%**. Further analysis revealed that the agent predominantly ended episodes by falling into holes. To alleviate this predicament, we attempted various conventional hyperparameter adjustments and policy improvements, including:

* **Increasing the number of agents:** Hoping to increase the probability of finding rewards through parallel exploration.
* **Increasing the minimum randomness (epsilon) of decisions and slowing its decay rate:** To encourage longer exploration.
* **Increasing the learning rate (alpha) and raising the discount factor (gamma):** Attempting to accelerate Q-value updates and the propagation of long-term rewards.
* **Increasing the maximum steps per episode (max\_len):** To give the agent more time to explore.

However, all these attempts **failed to mitigate the most fundamental and severe problem: reward sparsity**. The agent virtually never "saw" a reward throughout the entire training process, causing its Q-table to remain largely un-updated and preventing it from forming an effective decision-making policy. This clearly indicated the need for a radical overhaul of the reward mechanism.

### Phase 4: Custom Reward Engineering

To overcome the severe challenge of sparse rewards, we turned to **Reward Shaping**, meticulously designing a custom reward mechanism aimed at providing the agent with denser and more guiding feedback signals. We primarily focused on the following aspects:

1. **Significant Increase in Goal Reward (Goal Reward Amplification):**
   We drastically increased the reward for successfully reaching the goal ('G') from the default `1` to `10`. This modification was crucial as it ensured that the strength of the goal reward signal could **effectively penetrate the decay of the discount factor (gamma)**. This maintained its "attractiveness" throughout the Q-value update chain, preventing the reward signal from being prematurely "diluted" and thus ensuring its reliable back-propagation to distant states.

2. **Distance-Based Reward:**
   We introduced a dynamic distance reward to encourage the agent to continuously move closer to the goal. The core logic is that for every step the agent takes, if its **Manhattan distance** to the goal decreases, it receives a corresponding positive reward. The reward is calculated as follows:

   ```python
   def get_distance_reward(self, state):
       """Calculate distance reward"""
       row, col = divmod(state, 8)
       goal_row, goal_col = 7, 7 # Goal position is (7, 7)

       distance = abs(row - goal_row) + abs(col - goal_col) # Calculate Manhattan distance from current state to goal
       max_distance = 14 # Max Manhattan distance in an 8x8 grid is (7-0) + (7-0) = 14

       # The closer the distance, the higher the reward, but the reward value decreases as it gets closer to the goal
       distance_reward = (max_distance - distance) / max_distance * 0.1
       return distance_reward
   ```

   This design incorporates an important consideration: although the agent receives a higher "base" distance score as it gets closer to the goal, **by dividing by `max_distance` and multiplying by a small coefficient `0.1`, we ensure that the absolute value of the distance reward remains relatively small, and that the reward for each step forward gradually decreases as the agent approaches the goal.** The purpose of this is to prevent the agent from over-relying on distance rewards and engaging in "reward cheating" behavior (i.e., merely wandering to accumulate distance rewards), ensuring its ultimate objective remains reaching the final destination.

3. **Multi-layered Penalty System:**
   To guide the agent away from dangerous areas and useless actions, we designed robust penalty mechanisms:

   * **Fall into Hole Penalty (-1):** When the agent falls into a hole ('H'), it immediately receives a significant penalty of **-1**. This strong negative feedback is designed to quickly and clearly signal to the agent that these areas are absolute no-go zones, thereby forcing it to learn to avoid traps.
   * **Wall Collision Penalty (-0.5):** If the agent attempts an invalid action (e.g., trying to move into a map boundary), it receives a penalty of **-0.5**. This helps reduce unnecessary exploration and encourages the agent to select valid actions that actually change its state.

Through these custom reward designs, we successfully constructed a denser and more guiding learning environment for the agent. Initial experimental results were encouraging: in specific 8x8 map configurations with fewer holes and more viable paths, the agent was able to learn and reach the goal with remarkable ease. In the final tests for these scenarios, 20 agents using a greedy policy (i.e., strictly following learned Q-values) were all able to **reach the goal in the shortest 14-step path**, demonstrating the powerful guiding capability of custom rewards.

### Phase 5: Perfect Solution (14-step Optimal Path)

Although the reward shaping from Phase 4 achieved significant success in certain simpler configurations, new challenges emerged when faced with more demanding maps. We designed a **more difficult 8x8 map, where strategically placed holes confined the successful path to a single, winding route**, forcing the agent into complex turns and detours.

During training on this "hardcore" map, we observed a stubborn **local optimum problem**: the agent would **oscillate repeatedly between two states** at critical "turns," particularly before needing to turn from the bottom upwards into a new path. In-depth analysis revealed that this was due to the **limitations of our previously designed Manhattan distance reward**. When the agent was at a state just before a turn, and preparing to enter the state within the turn, even if that turn was correct in the actual path, the **Manhattan distance of states inside or even after the turn might be longer or not significantly change compared to the current state**. This caused the agent to be "penalized" by the distance reward, leading it to favor avoiding these seemingly "further away" areas. Instead, it chose to oscillate "grinding" for Manhattan distance rewards in areas where the distance change was minimal (i.e., moving back and forth at the bottom of the rectangle), thereby reinforcing the left-to-right movement rather than turning upwards.

To break this local loop and encourage the agent to explore states that might not offer an immediate Manhattan distance advantage but are part of the correct long-term path, we introduced a **Curiosity-Driven Exploration Bonus** as a form of intrinsic motivation. The core idea is that whenever the agent visits a **new or less-frequently visited state**, it receives an additional positive reward, thereby encouraging it to leave its "comfort zone" and explore unfamiliar areas of the environment.

Our designed exploration reward function `get_exploration_bonus` comprises several layers:

```python
    def get_exploration_bonus(self, state):
        """Calculate exploration reward"""
        total_bonus = 0.0

        # 1. First Visit Bonus (first visit within the current episode)
# Encourages the agent to explore more states within an episode, rather than repeatedly visiting known states
        if self.episode_visit_count[state] == 0:
            first_visit_bonus = self.exploration_bonus_strength
            total_bonus += first_visit_bonus

            # If it's a global first visit, grant an additional "curiosity" reward
            # Aims to encourage the agent to discover never-before-visited areas on the map
            if self.state_visit_count[state] == 0:
                total_bonus += self.curiosity_bonus
                print(f"🆕 Discovered new state {state}! Reward: +{self.curiosity_bonus}")

        # 2. Rarity Bonus (higher reward for less frequently visited states)
        # Encourages the agent to explore states that, even if visited in past episodes, have low overall visit frequency
        if self.state_visit_count[state] < self.novelty_threshold:
            rarity_bonus = self.exploration_bonus_strength / (1 + self.state_visit_count[state])
            total_bonus += rarity_bonus

        # 3. Time-decaying Exploration Reward
        # Gradually reduces the importance of exploration reward as training episodes progress, allowing the agent to rely more on external rewards
        decay_factor = (self.exploration_bonus_decay ** self.current_episode)
        total_bonus *= max(0.1, decay_factor)  # Set a minimum exploration reward to prevent premature disappearance

        return total_bonus
```

By incorporating this multi-layered exploration reward mechanism, the agent was no longer confined solely by short-term distance advantages. It became incentivized to explore new states, even if their Manhattan distance wasn't immediately "optimal." This intrinsic motivation successfully broke the previous "back-and-forth" deadlock, enabling the agent to bravely "enter the turn" and navigate complex paths. Ultimately, the agent, combining custom rewards with curiosity-driven exploration, was able to **successfully find and navigate to the goal, regardless of how winding and intricate the map path was, achieving a truly "perfect solution"**.

---
