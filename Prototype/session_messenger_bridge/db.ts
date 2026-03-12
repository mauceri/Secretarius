import { DataTypes, Sequelize } from "sequelize";

const storagePath = process.env.SESSION_MESSENGER_SQLITE_PATH || "session_messenger_messages.sqlite";

const sequelize = new Sequelize({
  dialect: "sqlite",
  storage: storagePath,
  logging: false,
});

const SessionMessage = sequelize.define("SessionMessage", {
  messageId: {
    type: DataTypes.STRING,
    allowNull: false,
    primaryKey: true,
  },
  timestamp: {
    type: DataTypes.DATE,
    allowNull: false,
    defaultValue: DataTypes.NOW,
  },
});

export { sequelize, SessionMessage };
