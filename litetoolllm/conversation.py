import json, os

def saveConversation(messages, directory, name):
  if not os.path.exists(directory):
    os.makedirs(directory)
  safeName = os.path.basename(name)
  path = os.path.join(directory, safeName + ".json")
  f = open(path, "w")
  json.dump(messages, f)
  f.close()
  print("saved conversation to " + path)
  return path

def trimHistory(messages, maxMessages):
  if len(messages) <= maxMessages:
    return messages
  if messages and messages[0].get("role") == "system":
    return [messages[0]] + messages[-(maxMessages - 1):]
  return messages[-maxMessages:]
