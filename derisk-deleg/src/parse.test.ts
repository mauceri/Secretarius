import { describe, it, expect } from "vitest";
import { parseReply } from "./parse";

describe("parseReply", () => {
  it("sépare l'id du corps", () => {
    expect(parseReply("18ab body de la réponse")).toEqual({
      messageId: "18ab",
      body: "body de la réponse",
    });
  });
  it("refuse sans corps", () => {
    expect(parseReply("18ab")).toBeNull();
    expect(parseReply("18ab   ")).toBeNull();
  });
  it("refuse une chaîne vide", () => {
    expect(parseReply("")).toBeNull();
  });
  it("conserve les espaces internes du corps", () => {
    expect(parseReply("X  deux  espaces")).toEqual({
      messageId: "X",
      body: "deux  espaces",
    });
  });
});
