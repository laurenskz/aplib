package nl.uu.cs.aplib.exampleUsages.fiveGame;

import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGame.GAMESTATUS;
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGame.SQUARE;
import nl.uu.cs.aplib.exampleUsages.fiveGame.FiveGame.Square_;
import nl.uu.cs.aplib.mainConcepts.Environment;
import nl.uu.cs.aplib.mainConcepts.Environment.EnvOperation;

public class FiveGameEnv extends Environment {

    static class FiveGameConf {
        private int boardSize;
        private int availableSpots;
        private boolean[][] blocked;

        public int getBoardSize() {
            return boardSize;
        }

        public boolean[][] getBlocked() {
            return blocked;
        }

        public int getAvailableSpots() {
            return availableSpots;
        }
    }


    private FiveGame thegame;

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
        conf = new FiveGameConf();
        setConfiguration();
        return this;
    }

    private void setConfiguration() {
        conf.boardSize = boardsize;
        conf.blocked = new boolean[boardsize][boardsize];
        for (int i = 0; i < boardsize; i++) {
            for (int j = 0; j < boardsize; j++) {
                conf.blocked[i][j] = board[i][j] == SQUARE.BLOCKED;
                if (!conf.blocked[i][j]) {
                    conf.availableSpots++;
                }
            }
        }
    }

    @Override
    public void refreshWorker() {
        lastmove = thegame.getLastmove();
    }

    @Override
    protected Object sendCommand_(EnvOperation cmd) {
        if (cmd.command.equals("move")) {
            Object[] arg_ = (Object[]) cmd.arg;
            thegame.move((SQUARE) arg_[0], (int) arg_[1], (int) arg_[2]);
            return thegame.getGameStatus();
        } else
            throw new IllegalArgumentException();
    }

    public GAMESTATUS move(SQUARE ty, int x, int y) {
        Object[] arg = {ty, (Integer) x, (Integer) y};
        var o = sendCommand("ANONYMOUS", null, "move", arg);
        return (GAMESTATUS) o;
    }

}
