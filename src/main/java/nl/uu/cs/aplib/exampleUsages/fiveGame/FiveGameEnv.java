package nl.uu.cs.aplib.exampleUsages.fiveGame;

import kotlin.Pair;
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGame.GAMESTATUS;
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGame.SQUARE;
import nl.uu.cs.aplib.mainConcepts.Environment;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

public class FiveGameEnv extends Environment {

    public static class FiveGameConf {
        private final int boardSize;
        private int availableSpots;
        private final boolean[][] blocked;
        private final List<Square_> spots;

        public FiveGameConf(int size, Predicate<Pair<Integer, Integer>> predicate) {
            this.boardSize = size;
            this.blocked = new boolean[size][size];
            this.spots = new ArrayList<>();
            for (int j = 0; j < size; j++) {
                for (int i = 0; i < size; i++) {
                    blocked[i][j] = predicate.test(new Pair<>(i, j));
                    if (!blocked[i][j]) {
                        availableSpots++;
                        spots.add(new Square_(SQUARE.EMPTY, i, j));
                    }
                }
            }
        }

        public int getBoardSize() {
            return boardSize;
        }

        public boolean[][] getBlocked() {
            return blocked;
        }

        public int getAvailableSpots() {
            return availableSpots;
        }

        public List<Square_> getSpots() {
            return spots;
        }

        public int spotPos(int x, int y) {
            for (int i = 0; i < spots.size(); i++) {
                if (x == spots.get(i).x && y == spots.get(i).y) return i;
            }
            throw new IllegalArgumentException("Not a spot:" + x + "," + y);
        }
    }


    FiveGame thegame;

    // variables for keeping track relevant part of FiveGame's state:
    int boardsize;
    SQUARE[][] board;
    Square_ lastmove;
    FiveGameConf conf;

    public FiveGameEnv() {
        super();
    }

    public FiveGameEnv attachGame(FiveGame g) {
        thegame = g;
        board = g.getState();
        boardsize = g.boardsize;
        setConfiguration();
        return this;
    }

    private void setConfiguration() {
        conf = new FiveGameConf(boardsize, pair -> board[pair.getFirst()][pair.getSecond()] == SQUARE.BLOCKED);

    }

    @Override
    protected Object sendCommand_(EnvOperation cmd) {
    	
    	switch (cmd.command) {
    	   case "move" : 
    		   Object[] arg_ = (Object[]) cmd.arg;
               thegame.move((SQUARE) arg_[0], (int) arg_[1], (int) arg_[2]);
               return thegame.getGameStatus();
    	   case "observe" : 
    		   return new Pair<GAMESTATUS, SQUARE[][]>(thegame.getGameStatus(), thegame.board) ;
    	}
    	throw new IllegalArgumentException();
    }
    
    @Override
    public Pair<GAMESTATUS, SQUARE[][]> observe(String agentId) {
        return (Pair<GAMESTATUS, SQUARE[][]>) this.sendCommand("ANONYMOUS", null, "observe", null) ;
    }


    public GAMESTATUS move(SQUARE ty, int x, int y) {
        Object[] arg = {ty, (Integer) x, (Integer) y};
        var o = sendCommand("ANONYMOUS", null, "move", arg);
        return (GAMESTATUS) o;
    }

    @Override
    public void resetWorker() {
        lastmove = null;

        thegame.reset();
    }

    @Override
    public String toString() {
        return thegame.toString();
    }
}
