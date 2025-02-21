---
name: Ticket
about: A ticket that relates to a tech spec requirement that will be refined during planning.
---

# [Title]

## Description

[Description of the ticket]

### Link to feature design or tech spec

[Provide links or documents here for context.]

### Stakeholders

[Please reference the stakeholders for this ticket]

### Functional Requirements (FR)

What does the ticket need to do? What are the steps to implement the requirements? This should be as detailed as needed for anyone to be able to contribute

Example:

1. Create a new route `POST /user` in the `routes/user/router.py` file to create a new user.
2. Add the `UserCreate` schema to the `schemas.py` file.
3. The route should take the `UserCreate` schema as the `POST` body.
4. The initial type validation should be handled by the pydantic model/schema.
5. A new `create` method should be added to the `UserService`.
   1. The `create` method should check to see if the user already exists by `email` by using the existing `exists` method on `UserService`
   2. If a user exists a `DuplicateUserException` should be raised.
   3. Otherwise, create the user in the db by converting the `UserCreate` schema into the `User` db model.
   4. Return the newly created user.
6. Return a `200` HTTP status along with the newly created user.

### Non-functional Requirements (NFR)

What constraints exist for this ticket? All constraints must be quantifiable.

Example:

- Creating a user does not take longer than 1 second.

### Acceptance Criteria

What needs to happen functionally for this work to pass review? Usually a combination of FRs and NFRs as test cases written in plain text. This should follow test description formats. "Should...when..."

Example:

1. Should return a status code of `200` when a new user is created successfully.
2. Should return new user information when a user is created successfully.
3. Should raise a `422` exception when an email is not sent in the body.
4. Should raise a `422` exception when an email is not correctly formatted.
5. Should raise a `409` exception when a user already exists with the provided `email`.
