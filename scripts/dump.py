        # Agent is brushing against the walls
        # First check for a net positive effect of the wind
        if abs( (newCol - self.agentCol) ) + abs( (newRow - self.agentRow) ) == 2:
            if newRow == 0: newRow += 1                         # Check for top wall
            if newRow == len(self.map) - 1: newRow -= 1         # Check for bottom wall
            if newCol >= len(self.map[0]) - 1:  newCol -= 1     # Check for right wall

